import re
import argparse
import pandas as pd
from pathlib import Path

# Stała konwersji Hartree -> kcal/mol
EH2KCAL = 627.5095 
# Stała konwersji Hartree -> kJ/mol
EH2KJ = 2625.49953


def find_gibbs_energy(file_path):
    """
    Wyciąga energię Gibbsa z pliku.
    Zwraca wartość z OSTATNIEJ znalezionej linijki pasującej do wzorca.
    """
    pattern = re.compile(r"Final Gibbs free energy\s+\.\.\.\s+([-?\d.]+)\s+Eh")
    last_energy = None  
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    last_energy = float(match.group(1))
                    
        return last_energy

    except Exception:
        return None


def collect_files(paths):
    """Zbiera pliki z podanych ścieżek."""
    all_files = []
    for path_str in paths:
        p = Path(path_str)
        if p.is_dir():
            all_files.extend(p.glob('*'))
        elif p.is_file():
            all_files.append(p)
    
    return all_files


def process_files(file_paths):
    """Przetwarza pliki i zwraca DataFrame z wynikami."""
    results = []
    
    for file_path in file_paths:
        energy = find_gibbs_energy(file_path)
        if energy is not None:
            results.append({
                "Filename": file_path.name,
                "Path": str(file_path.absolute()),
                "E [Eh]": energy
            })
    
    if not results:
        return None
    
    # Tworzenie DataFrame
    df = pd.DataFrame(results)
    
    # Dodanie kolumny z konwersją na kcal/mol
    df["E [kcal/mol]"] = df["E [Eh]"] * EH2KCAL
    df["E [kJ/mol]"] = df["E [Eh]"] * EH2KJ
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Wyciąga OSTATNIĄ Gibbs Free Energy do ramki danych (Pandas)."
    )
    parser.add_argument("paths", nargs='+', help="Ścieżki do plików lub folderów")
    parser.add_argument("-o", "--output", help="Opcjonalna nazwa pliku CSV do zapisu")
    
    args = parser.parse_args()
    
    # Zbieranie i przetwarzanie plików
    all_files = collect_files(args.paths)
    df = process_files(all_files)
    
    # Wyświetlanie wyników
    if df is not None:
        print("\n--- Znalezione wyniki (ostatnie wystąpienie w pliku) ---")
        print(df.to_string(index=False))
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n[INFO] Wyniki zapisano do pliku: {args.output}")
    else:
        print("Nie znaleziono pasujących danych w podanych ścieżkach.")


if __name__ == "__main__":
    main()
