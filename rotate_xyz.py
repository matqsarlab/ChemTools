import argparse
import sys
import numpy as np
from typing import List, Tuple, Optional


class Quaternion:
    """Minimalistyczna obsługa kwaternionów do rotacji wektorów."""

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w, self.x, self.y, self.z = w, x, y, z

    def normalize(self) -> "Quaternion":
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm == 0:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle_deg: float) -> "Quaternion":
        angle_rad = np.radians(angle_deg)

        # Zabezpieczenie: upewniamy się, że axis ma kształt (3,)
        if axis.shape != (3,):
            raise ValueError(f"Oczekiwano wektora 3D, otrzymano kształt {axis.shape}")

        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-8:
            return Quaternion(1.0, 0.0, 0.0, 0.0)

        ux, uy, uz = axis / axis_len
        half_theta = angle_rad / 2
        sin_half = np.sin(half_theta)

        return Quaternion(
            np.cos(half_theta), ux * sin_half, uy * sin_half, uz * sin_half
        ).normalize()


class MoleculeRotator:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.atoms: List[str] = []
        self.coords: np.ndarray = np.empty((0, 3))
        self.header: List[str] = []
        self._load_xyz()

    def _load_xyz(self):
        try:
            with open(self.input_file, "r") as f:
                lines = f.readlines()

            self.header = lines[:2]
            atoms_list = []
            coords_list = []

            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    atoms_list.append(parts[0])
                    coords_list.append([float(x) for x in parts[1:4]])

            self.atoms = atoms_list
            self.coords = np.array(coords_list)

        except Exception as e:
            print(f"Błąd wczytywania pliku: {e}", file=sys.stderr)
            sys.exit(1)

    def parse_axis(self, axis_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parsuje string wejściowy do postaci (origin, axis_vector).
        Jest teraz bardziej odporny na formatowanie i obsługuje indeksy atomów.
        """
        # Zamieniamy wszystko co nie jest liczbą lub minusem/kropką na przecinek
        clean_str = (
            axis_str.replace(";", ",")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "")
        )

        try:
            # Rozbijamy po przecinkach i filtrujemy puste stringi
            values = [float(x) for x in clean_str.split(",") if x]

            if len(values) == 2:
                # Mamy 2 liczby -> indeksy atomów (numeracja od 0)
                idx1, idx2 = int(values[0]), int(values[1])

                # Sprawdzenie, czy indeksy nie wychodzą poza zakres
                if not (0 <= idx1 < len(self.coords)) or not (
                    0 <= idx2 < len(self.coords)
                ):
                    raise ValueError(
                        f"Indeksy atomów poza zakresem (0 do {len(self.coords)-1})"
                    )

                p1 = self.coords[idx1]
                p2 = self.coords[idx2]
                return p1, p2 - p1  # p1 to origin, (p2 - p1) to wektor kierunkowy

            elif len(values) == 6:
                # Mamy 6 liczb -> dwa punkty (x2,y2,z2) i (x1,y1,z1)
                p2 = np.array(values[0:3])
                p1 = np.array(values[3:6])
                return p1, p2 - p1

            elif len(values) == 3:
                # Mamy 3 liczby -> wektor kierunkowy, origin to 0,0,0
                vec = np.array(values)
                return np.zeros(3), vec

            else:
                raise ValueError(
                    f"Oczekiwano 2, 3 lub 6 liczb, otrzymano {len(values)}"
                )

        except ValueError as e:
            print(
                f"Błąd formatu osi: {e}.\nUżyj np: '1.0,0,0' (kierunek), '1,0,0;0,0,0' (2 punkty 3D) lub '0,5' (2 atomy)",
                file=sys.stderr,
            )
            sys.exit(1)

    def rotate(self, axis_str: str, selection_indices: List[int], angle_deg: float):
        if self.coords.size == 0:
            return

        origin, axis_vec = self.parse_axis(axis_str)

        # Upewniamy się, że wektor osi jest poprawny dla from_axis_angle
        try:
            q_rot = Quaternion.from_axis_angle(axis_vec, angle_deg)
        except ValueError as e:
            print(f"Błąd obliczeń kwaternionu: {e}", file=sys.stderr)
            sys.exit(1)

        q_rot_conj = q_rot.conjugate()

        for idx in selection_indices:
            if idx < 0 or idx >= len(self.coords):
                continue

            v = self.coords[idx] - origin

            p_quat = Quaternion(0, v[0], v[1], v[2])
            rotated_quat = q_rot.multiply(p_quat).multiply(q_rot_conj)

            self.coords[idx] = (
                np.array([rotated_quat.x, rotated_quat.y, rotated_quat.z]) + origin
            )

    def output(self, output_file: Optional[str] = None):
        res = []
        if self.header:
            res.append(self.header[0].strip())
            res.append(self.header[1].strip() + " [Rotated]")

        for atom, c in zip(self.atoms, self.coords):
            res.append(f"{atom:<2}    {c[0]:14.10f}    {c[1]:14.10f}    {c[2]:14.10f}")

        text = "\n".join(res)

        if output_file:
            with open(output_file, "w") as f:
                f.write(text)
        else:
            print(text)


def main():
    parser = argparse.ArgumentParser(description="Rotacja plików XYZ kwaternionami.")
    parser.add_argument("-i", "--input", required=True, help="Plik wejściowy XYZ")
    parser.add_argument("-o", "--output", help="Plik wyjściowy (opcjonalny)")
    parser.add_argument(
        "-a",
        "--axis",
        required=True,
        help="Oś: 2 indeksy atomów (np. '0,5'), wektor 'x,y,z' lub dwa punkty 'x2,y2,z2;x1,y1,z1'",
    )
    parser.add_argument(
        "-s", "--select", required=True, help="Indeksy atomów (np. '0,1,2' lub 'all')"
    )
    parser.add_argument(
        "-d", "--deg", type=float, default=180.0, help="Kąt w stopniach"
    )

    args = parser.parse_args()

    rotator = MoleculeRotator(args.input)

    if args.select.lower() == "all":
        indices = list(range(len(rotator.coords)))
    else:
        # Zamiana przecinków na spacje, aby obsłużyć "1,2,3" i "1 2 3"
        s = args.select.replace(",", " ")
        indices = [int(x) for x in s.split()]

    rotator.rotate(args.axis, indices, args.deg)
    rotator.output(args.output)


if __name__ == "__main__":
    main()
