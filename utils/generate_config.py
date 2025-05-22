import argparse
import random

def generate_config_file(filename, num_epochs, num_particles):
    with open(filename, 'w') as f:
        # Riga 1: numero di epoche
        f.write(f"{num_epochs}\n")

        # Riga 2: numero di particelle
        f.write(f"{num_particles}\n")

        # Riga 3+: per ogni particella: posizione x, y, z - velocità x, y, z - massa
        for _ in range(num_particles):
            position = [random.uniform(-100.0, 100.0) for _ in range(3)]
            velocity = [random.uniform(-10.0, 10.0) for _ in range(3)]
            mass = random.uniform(0.1, 100.0)
            values = position + velocity + [mass]
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Genera un file di configurazione per la simulazione.")
    parser.add_argument('--epochs', '-e', type=int, required=True, help="Numero di epoche")
    parser.add_argument('--particles', '-p', type=int, required=True, help="Numero di particelle")
    parser.add_argument('--output', '-o', default="config.txt", help="Nome del file di output (default: config.txt)")

    args = parser.parse_args()

    generate_config_file(args.output, args.epochs, args.particles)
    print(f"File '{args.output}' generato con successo.")

if __name__ == "__main__":
    main()

