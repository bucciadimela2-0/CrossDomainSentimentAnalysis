import pandas as pd
import matplotlib.pyplot as plt
domains = ['Birth Control', 'Depression', 'Pain', 'Anxiety', 'Diabetes, Type 2']

# Creiamo un DataFrame vuoto con i domini come colonne e indici
df = pd.DataFrame(index=domains, columns=domains)

# Inizializziamo la diagonale con il valore 'Accuracy' e "Cohen's Kappa"
for domain in domains:
    df.at[domain, domain] = ['Accuracy: 1.0', f"Cohen's Kappa: 1.0"]

# Popoliamo il DataFrame con i valori forniti
data = [
    ('Birth Control', 'Birth Control', 0.919051, 0.859419),
    ('Birth Control', 'Depression', 0.489176, 0.138806),
    ('Birth Control', 'Pain', 0.641905, 0.194697),
    ('Birth Control', 'Anxiety', 0.487421, 0.117242),
    ('Birth Control', 'Diabetes, Type 2', 0.690594, 0.352564),
    ('Depression', 'Birth Control', 0.550477, 0.094754),
    ('Depression', 'Depression', 0.907593, 0.779572),
    ('Depression', 'Pain', 0.780476, 0.218655),
    ('Depression', 'Anxiety', 0.785639, 0.190575),
    ('Depression', 'Diabetes, Type 2', 0.706683, 0.217505),
    ('Pain', 'Birth Control', 0.522077, 0.020739),
    ('Pain', 'Depression', 0.705977, 0.083465),
    ('Pain', 'Pain', 0.896667, 0.679238),
    ('Pain', 'Anxiety', 0.769916, 0.087551),
    ('Pain', 'Diabetes, Type 2', 0.668317, 0.045157),
    ('Anxiety', 'Birth Control', 0.522595, 0.021403),
    ('Anxiety', 'Depression', 0.710824, 0.096146),
    ('Anxiety', 'Pain', 0.773810, 0.095927),
    ('Anxiety', 'Anxiety', 0.898847, 0.689079),
    ('Anxiety', 'Diabetes, Type 2', 0.684406, 0.092854),
    ('Diabetes, Type 2', 'Birth Control', 0.593595, 0.220115),
    ('Diabetes, Type 2', 'Depression', 0.685299, 0.270467),
    ('Diabetes, Type 2', 'Pain', 0.661905, 0.155624),
    ('Diabetes, Type 2', 'Anxiety', 0.703354, 0.220188),
    ('Diabetes, Type 2', 'Diabetes, Type 2', 0.935644, 0.861950)
]
for item in data:
    domain_origin, domain_destination, accuracy, kappa = item
    df.at[domain_origin, domain_destination] = [f'Accuracy: {accuracy}', f"Cohen's Kappa: {kappa}"]

# Funzione per estrarre il valore numerico dall'accuratezza
def extract_accuracy(value):
    return float(value[0].split()[1])

# Calcola le medie per le colonne e le righe basate su Accuracy
column_means = df.applymap(extract_accuracy).mean()
row_means = df.applymap(extract_accuracy).mean(axis=1)

# Aggiungi le colonne e le righe delle medie al DataFrame
df['Column Means'] = column_means
df.loc['Row Means'] = row_means

# Stampa il DataFrame
print(df.to_string())

# Genera un grafico a tabella dal DataFrame
fig, ax = plt.subplots(figsize=(10, 6))  # Imposta le dimensioni del grafico
ax.axis('tight')  # Imposta gli assi come 'tight' per adattarsi al contenuto
ax.axis('off')  # Nasconde gli assi

# Crea una tabella dalla DataFrame e aggiungila al grafico
table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')

# Imposta lo stile della tabella
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(2.0, 2.0)  # Imposta la scala per migliorare la leggibilità

# Salva il grafico come immagine PNG
plt.savefig('table.png', bbox_inches='tight', pad_inches=0.1)






