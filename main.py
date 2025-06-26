import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

EXPERIMENTOS = 20
POP = 100
GERACOES = 40

TAX_CROSSOVER = 0.65
TAX_MUT = 0.08

TAM_CROM = 22
TOTAL_BITS = TAM_CROM * 2

NORM_MIN, NORM_MAX = 1, 100
MIN_VAL, MAX_VAL = -100, 100

def f6(x, y):
    r2 = x**2 + y**2
    return 0.5 - ((np.sin(np.sqrt(r2))**2 - 0.5) / (1 + 0.001 * r2)**2)

def bin_para_real(bits):
    x_bin = int(bits, 2)
    fator = (MAX_VAL - MIN_VAL) / (2**TAM_CROM - 1)
    x_real = x_bin * fator + MIN_VAL
    return x_real

def gerar_populacao():
    return [''.join(np.random.choice(['0', '1'], size=TOTAL_BITS)) for _ in range(POP)]

def decodificar(pop):
    reais = []
    for crom in pop:
        x_bin = crom[:TAM_CROM]
        y_bin = crom[TAM_CROM:]
        x_real = bin_para_real(x_bin)
        y_real = bin_para_real(y_bin)
        reais.append((x_real, y_real))
    return np.array(reais)

def avaliar_padrao(pop):
    reais = decodificar(pop)
    f_vals = f6(reais[:, 0], reais[:, 1])
    apt = f_vals
    return f_vals, apt

def avaliar_normalizado(pop):
    reais = decodificar(pop)
    f_vals = f6(reais[:, 0], reais[:, 1])
    ranks = f_vals.argsort().argsort()
    apt = NORM_MIN + ((NORM_MAX-NORM_MIN) / (POP - 1))*(ranks)
    return f_vals, apt

def mutacao(crom, taxa=TAX_MUT):
    return ''.join(bit if np.random.rand() > taxa else '1' if bit == '0' else '0' for bit in crom)

def crossover(p1, p2, taxa=TAX_CROSSOVER):
    if np.random.rand() < taxa:
        ponto = np.random.randint(1, TOTAL_BITS - 1)
        return p1[:ponto] + p2[ponto:], p2[:ponto] + p1[ponto:]
    return p1, p2

def selecao(pop, apt):
    prob = apt / apt.sum()
    idx = np.random.choice(len(pop), size=2, replace=False, p=prob)
    return pop[idx[0]], pop[idx[1]]

def contar_noves(num):
    getcontext().prec = 50
    d = Decimal(str(num)).normalize() 
    decimal_str = str(d).split('.')

    if len(decimal_str) < 2:
        return 0 

    dec = decimal_str[1]
    if not dec.startswith('9'):
        return 0  

    count = 0
    for c in dec:
        if c == '9':
            count += 1
        else:
            break
    return count

def ga_geracional(elitismo=True, normalizado=False):
    pop = gerar_populacao()
    melhores_por_geracao = []
    noves_por_geracao = []

    for _ in range(GERACOES):
        avaliar = avaliar_normalizado if normalizado else avaliar_padrao
        f_vals, apt = avaliar(pop)

        if elitismo:
            elite_idx = np.argmax(f_vals)
            elite = pop[elite_idx]

        nova_pop = []
        while len(nova_pop) < POP:
            p1, p2 = selecao(pop, apt)
            f1, f2 = crossover(p1, p2)
            nova_pop.append(mutacao(f1))
            if len(nova_pop) < POP:
                nova_pop.append(mutacao(f2))

        pop = nova_pop[:POP]

        if elitismo:
            worst_idx = np.argmin(avaliar(pop)[0])
            pop[worst_idx] = elite

        melhor_f = max(f_vals)
        melhores_por_geracao.append(melhor_f)
        noves_por_geracao.append(contar_noves(melhor_f))

    return melhores_por_geracao, noves_por_geracao

def ga_steady_state(gap=0.1):
    pop = gerar_populacao()
    melhores_por_geracao = []
    noves_por_geracao = []

    for _ in range(GERACOES):
        f_vals, apt = avaliar_normalizado(pop)
        novos_filhos = []
        num_filhos = int(gap * POP)
        
        elite_idx = np.argmax(f_vals)
        elite = pop[elite_idx]

        while len(novos_filhos) < num_filhos:
            p1, p2 = selecao(pop, apt)
            f1, f2 = crossover(p1, p2)
            filhos = [mutacao(f1), mutacao(f2)]
            novos_filhos.extend(filhos)

        filhos_f = avaliar_normalizado(novos_filhos)[0]
        piores_idxs = np.argsort(apt)[:num_filhos]

        # Substituir os piores pelos filhos (melhores primeiros)
        for idx, filho in zip(piores_idxs, novos_filhos):
            pop[idx] = filho
            
        worst_idx = np.argmin(avaliar_normalizado(pop)[0])
        pop[worst_idx] = elite

 
        melhor_f = max(avaliar_normalizado(pop)[0])
        melhores_por_geracao.append(melhor_f)
        noves_por_geracao.append(contar_noves(melhor_f))

    return melhores_por_geracao,noves_por_geracao


def executar_experimentos_todos(funcao_ga, label):
    resultados_melhores = []
    resultados_noves = []
    for _ in range(EXPERIMENTOS):
        melhores, noves = funcao_ga()
        resultados_melhores.append(melhores)
        resultados_noves.append(noves)
    return label, resultados_melhores, resultados_noves

experimentos = [
    (lambda: ga_geracional(elitismo=False, normalizado=False), "Sem elitismo + Avaliação simples"),
    (lambda: ga_geracional(elitismo=True, normalizado=False), "Elitismo + Avaliação simples"),
    (lambda: ga_geracional(elitismo=True, normalizado=True), "Elitismo + Normalizado"),
    (lambda: ga_steady_state(0.1), "Elitismo + Normalizado (Steady State gap = 0.1)"),
    (lambda: ga_steady_state(0.5), "Elitismo + Normalizado (Steady State gap = 0.5)"),
    (lambda: ga_steady_state(0.9), "Elitismo + Normalizado (Steady State gap = 0.9)")
]

resultados_todos = [executar_experimentos_todos(f, label) for f, label in experimentos]

#Gráfico 1: Média por experimento 
plt.figure(figsize=(12, 6))
for label, resultados_melhores, _ in resultados_todos:
    medias_por_experimento = [np.mean(r) for r in resultados_melhores]
    plt.plot(range(1, EXPERIMENTOS + 1), medias_por_experimento, label=label)

plt.title("Média da melhor solução por experimento")
plt.xlabel("Número do experimento")
plt.ylabel("Média da melhor solução (todas as gerações)")
plt.grid(True)
plt.legend()
plt.show()


#Gráfico 2: Número médio de 9 consecutivos por geração 
plt.figure(figsize=(12, 6))
for label, _, resultados_noves in resultados_todos:
    media_por_geracao = np.mean(resultados_noves, axis=0)
    plt.plot(range(1, GERACOES + 1), media_por_geracao, label=label)

plt.title("Número médio de 9 consecutivos fitness por geração")
plt.xlabel("Geração")
plt.ylabel("Número médio de 9s consecutivos")
plt.grid(True)
plt.legend()
plt.xticks(range(0, GERACOES + 1, 2))
plt.show()



