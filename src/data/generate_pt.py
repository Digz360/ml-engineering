import random
from pathlib import Path


random.seed(42)

OUTPUT_DIR = Path("src/data/llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

PT_PARAGRAPHS = 120_000
MATH_EXAMPLES = 50_000
CHAT_EXAMPLES = 30_000


# ============================================================
# PORTUGUÊS GERAL
# ============================================================

subjects = [
    "um estudante",
    "uma engenheira",
    "um programador",
    "uma professora",
    "um investigador",
    "uma família",
    "uma empresa",
    "uma equipa",
    "um atleta",
    "um médico",
    "uma criança",
    "um cientista",
    "um agricultor",
    "um cliente",
    "um viajante",
]

actions = [
    "organizar melhor o trabalho",
    "resolver um problema complexo",
    "aprender uma nova competência",
    "preparar uma apresentação",
    "analisar os resultados",
    "tomar uma decisão importante",
    "planear as tarefas da semana",
    "melhorar um processo",
    "compreender uma ideia nova",
    "comparar diferentes alternativas",
    "reduzir o tempo necessário",
    "encontrar uma solução prática",
]

places = [
    "em casa",
    "na escola",
    "na universidade",
    "no escritório",
    "num laboratório",
    "num hospital",
    "numa biblioteca",
    "numa cidade",
    "numa pequena empresa",
    "durante uma viagem",
]

consequences = [
    "obter melhores resultados",
    "reduzir erros",
    "poupar tempo",
    "compreender melhor o problema",
    "trabalhar de forma mais eficiente",
    "tomar uma decisão mais informada",
    "identificar possíveis dificuldades",
    "acompanhar o progresso",
    "melhorar a qualidade do trabalho",
]

concepts = {
    "computador":
        "uma máquina capaz de receber dados, executar instruções e produzir resultados",

    "algoritmo":
        "uma sequência organizada de passos usada para resolver um problema",

    "programa":
        "um conjunto de instruções que um computador consegue executar",

    "rede neuronal":
        "um modelo computacional composto por unidades ligadas que transformam informação através de várias camadas",

    "energia":
        "uma grandeza associada à capacidade de produzir movimento, calor ou outras transformações",

    "ecossistema":
        "um conjunto formado pelos organismos de uma região, pelo ambiente e pelas relações entre ambos",

    "DNA":
        "uma molécula que contém informação genética utilizada pelos organismos",

    "gravidade":
        "uma interação física que faz com que corpos com massa se atraiam",

    "Internet":
        "uma rede mundial que permite a comunicação entre computadores e outros dispositivos",

    "base de dados":
        "uma estrutura utilizada para armazenar e organizar informação",

    "sistema operativo":
        "o software responsável por gerir os recursos principais de um computador",

    "memória RAM":
        "uma memória rápida utilizada para guardar temporariamente dados necessários aos programas",

    "processador":
        "o componente responsável por executar grande parte das instruções de um computador",

    "fotossíntese":
        "um processo através do qual plantas e outros organismos transformam energia luminosa em energia química",

    "democracia":
        "um sistema político em que os cidadãos participam na escolha dos seus representantes",

    "inflação":
        "um aumento generalizado do nível dos preços ao longo do tempo",

    "proteína":
        "uma molécula formada por aminoácidos que desempenha várias funções importantes nos organismos",

    "átomo":
        "uma unidade fundamental da matéria constituída por partículas subatómicas",

    "planeta":
        "um corpo celeste que orbita uma estrela e possui massa suficiente para assumir uma forma aproximadamente esférica",

    "oceano":
        "uma grande massa de água salgada que cobre uma parte significativa da superfície terrestre",
}


countries = {
    "Portugal": "Europa",
    "Espanha": "Europa",
    "França": "Europa",
    "Alemanha": "Europa",
    "Itália": "Europa",
    "Brasil": "América do Sul",
    "Argentina": "América do Sul",
    "Canadá": "América do Norte",
    "Estados Unidos": "América do Norte",
    "Japão": "Ásia",
    "China": "Ásia",
    "Índia": "Ásia",
    "Egito": "África",
    "Angola": "África",
    "Austrália": "Oceânia",
}


def general_paragraph():
    subject = random.choice(subjects)
    action = random.choice(actions)
    place = random.choice(places)
    consequence = random.choice(consequences)

    templates = [
        (
            f"{subject.capitalize()} pretende {action} {place}. "
            f"Para isso, pode começar por identificar o objetivo principal e dividir o problema "
            f"em tarefas menores. Esta abordagem permite {consequence}."
        ),

        (
            f"Quando {subject} precisa de {action}, é útil analisar primeiro a situação. "
            f"Depois de compreender as opções disponíveis, torna-se mais fácil escolher os passos "
            f"necessários e {consequence}."
        ),

        (
            f"Uma forma prática de {action} consiste em definir prioridades, reunir a informação "
            f"necessária e executar cada etapa de forma organizada. "
            f"Este método pode ajudar {subject} a {consequence}."
        ),

        (
            f"{subject.capitalize()} encontra-se {place} e precisa de {action}. "
            f"Em vez de tentar resolver tudo ao mesmo tempo, pode começar pela tarefa mais importante, "
            f"avaliar o resultado e ajustar o plano quando necessário."
        ),
    ]

    return random.choice(templates)


def concept_paragraph():
    concept, definition = random.choice(list(concepts.items()))

    templates = [
        (
            f"{concept.capitalize()} é {definition}. "
            f"Compreender este conceito ajuda a perceber como diferentes componentes podem interagir "
            f"dentro de um sistema."
        ),

        (
            f"O conceito de {concept} pode ser explicado de forma simples: trata-se de {definition}. "
            f"A sua importância depende do contexto em que é utilizado."
        ),

        (
            f"Para compreender {concept}, é útil começar pela ideia principal. "
            f"Podemos descrevê-lo como {definition}. "
            f"Depois desta base, torna-se mais fácil estudar exemplos e aplicações concretas."
        ),
    ]

    return random.choice(templates)


def geography_paragraph():
    country, continent = random.choice(list(countries.items()))

    templates = [
        f"{country} é um país localizado no continente {continent}.",
        f"O país {country} encontra-se em {continent}.",
        f"Geograficamente, {country} pertence ao continente {continent}.",
    ]

    return random.choice(templates)


def causal_paragraph():
    causes = [
        ("a temperatura aumenta", "alguns materiais podem expandir-se"),
        ("uma pessoa pratica regularmente", "a execução de uma tarefa pode tornar-se mais fácil"),
        ("um computador recebe mais memória RAM", "pode manter mais informação disponível rapidamente"),
        ("uma planta recebe luz suficiente", "pode realizar fotossíntese"),
        ("uma equipa comunica claramente", "os membros conseguem coordenar melhor o trabalho"),
        ("a procura de um produto aumenta muito", "o preço pode aumentar se a oferta permanecer limitada"),
        ("um programa executa trabalho desnecessário", "pode demorar mais tempo a terminar"),
        ("uma pessoa dorme pouco", "a concentração pode diminuir no dia seguinte"),
    ]

    cause, effect = random.choice(causes)

    return (
        f"Existe uma relação entre causa e consequência. "
        f"Quando {cause}, {effect}. "
        f"Isto não significa que o resultado seja sempre garantido, porque outras condições "
        f"também podem influenciar a situação."
    )


def comparison_paragraph():
    pairs = [
        ("uma bicicleta", "um automóvel", "transporte"),
        ("memória RAM", "armazenamento", "computadores"),
        ("treino", "validação", "machine learning"),
        ("CPU", "GPU", "computação"),
        ("livro físico", "livro digital", "leitura"),
        ("trabalho individual", "trabalho em equipa", "organização"),
    ]

    a, b, context = random.choice(pairs)

    return (
        f"{a.capitalize()} e {b} podem ser comparados no contexto de {context}. "
        f"Cada opção apresenta características diferentes, por isso a escolha depende do objetivo, "
        f"dos recursos disponíveis e das prioridades de quem decide."
    )


def story_paragraph():
    names = [
        "Marta", "Pedro", "Ana", "Rui", "Joana",
        "Miguel", "Sofia", "Tiago", "Inês", "Carlos"
    ]

    name = random.choice(names)
    action = random.choice(actions)
    consequence = random.choice(consequences)

    return (
        f"{name} tinha uma tarefa para concluir e decidiu {action}. "
        f"Primeiro analisou aquilo de que precisava. Depois preparou os recursos e começou pela parte "
        f"mais importante. Ao acompanhar os resultados ao longo do processo, conseguiu {consequence}."
    )


def generate_pt():
    path = OUTPUT_DIR / "pt_v3.txt"

    generators = [
        general_paragraph,
        concept_paragraph,
        geography_paragraph,
        causal_paragraph,
        comparison_paragraph,
        story_paragraph,
    ]

    word_count = 0

    with path.open("w", encoding="utf-8") as f:
        for _ in range(PT_PARAGRAPHS):
            paragraph = random.choice(generators)()

            f.write(paragraph)
            f.write("\n\n")

            word_count += len(paragraph.split())

    print(f"PT V3: {word_count:,} palavras")


# ============================================================
# ARITMÉTICA
# ============================================================

def arithmetic_example():

    operation = random.choice(["+", "-", "*"])

    if operation == "+":
        a = random.randint(0, 1000)
        b = random.randint(0, 1000)
        result = a + b

    elif operation == "-":
        a = random.randint(0, 1000)
        b = random.randint(0, a)
        result = a - b

    else:
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        result = a * b

    templates = [
        f"{a} {operation} {b} = {result}.",

        f"O resultado de {a} {operation} {b} é {result}.",

        f"Calcula {a} {operation} {b}. "
        f"O resultado é {result}.",

        f"Utilizador: Quanto é {a} {operation} {b}? "
        f"Assistente: O resultado é {result}.",

        f"Utilizador: Calcula {a} {operation} {b}. "
        f"Assistente: {a} {operation} {b} = {result}.",
    ]

    return random.choice(templates)


def word_problem():
    a = random.randint(1, 100)
    b = random.randint(1, 100)

    templates = [
        (
            f"Ana tem {a} livros e recebe mais {b}. "
            f"Agora tem {a + b} livros."
        ),

        (
            f"Uma caixa contém {a} objetos e são adicionados mais {b}. "
            f"No total ficam {a + b} objetos."
        ),

        (
            f"Pedro tinha {a + b} euros e gastou {b} euros. "
            f"Ficou com {a} euros."
        ),

        (
            f"Existem {a} grupos com {b} elementos em cada grupo. "
            f"No total existem {a * b} elementos."
        ),
    ]

    return random.choice(templates)


def generate_math():
    path = OUTPUT_DIR / "aritmetica_v3.txt"

    with path.open("w", encoding="utf-8") as f:
        for _ in range(MATH_EXAMPLES):

            if random.random() < 0.75:
                example = arithmetic_example()
            else:
                example = word_problem()

            f.write(example)
            f.write("\n")

    print(f"Aritmética V3: {MATH_EXAMPLES:,} exemplos")


# ============================================================
# CHAT
# ============================================================

chat_questions = [
    (
        "O que é um algoritmo?",
        "Um algoritmo é uma sequência organizada de passos utilizada para resolver um problema ou executar uma tarefa."
    ),
    (
        "O que é uma rede neuronal?",
        "Uma rede neuronal é um modelo computacional formado por várias unidades ligadas em camadas que transformam informação e aprendem padrões a partir de dados."
    ),
    (
        "O que é inteligência artificial?",
        "Inteligência artificial é uma área da computação dedicada à criação de sistemas capazes de realizar tarefas associadas a capacidades como aprendizagem, previsão ou tomada de decisão."
    ),
    (
        "O que é memória RAM?",
        "A memória RAM guarda temporariamente dados que os programas precisam de utilizar rapidamente enquanto o computador está ligado."
    ),
    (
        "Qual é a diferença entre CPU e GPU?",
        "A CPU é um processador geral otimizado para executar vários tipos de tarefas, enquanto a GPU possui muitas unidades de cálculo e é especialmente eficiente em operações altamente paralelas."
    ),
    (
        "Portugal fica em que continente?",
        "Portugal fica na Europa."
    ),
    (
        "Brasil fica em que continente?",
        "O Brasil fica na América do Sul."
    ),
    (
        "O que é gravidade?",
        "A gravidade é uma interação física que provoca atração entre corpos com massa."
    ),
    (
        "O que é DNA?",
        "O DNA é uma molécula que contém informação genética utilizada no desenvolvimento e funcionamento dos organismos."
    ),
    (
        "O que é fotossíntese?",
        "A fotossíntese é o processo através do qual plantas e outros organismos utilizam luz para produzir energia química."
    ),
    (
        "Como posso organizar melhor o meu dia?",
        "Podes começar por identificar as tarefas mais importantes, definir prioridades e reservar períodos específicos para cada uma."
    ),
    (
        "Como posso aprender uma coisa nova?",
        "Começa pelos conceitos fundamentais, pratica com exemplos simples e aumenta gradualmente a dificuldade à medida que compreendes melhor o tema."
    ),
    (
        "Como posso resolver um problema complicado?",
        "Uma boa estratégia é dividir o problema em partes menores, analisar cada parte separadamente e depois combinar as soluções."
    ),
    (
        "Porque é importante dormir?",
        "Dormir permite ao organismo recuperar e desempenha um papel importante na memória, concentração e regulação de vários processos biológicos."
    ),
    (
        "O que é uma base de dados?",
        "Uma base de dados é uma estrutura utilizada para armazenar, organizar e consultar informação."
    ),
    (
        "O que faz um sistema operativo?",
        "Um sistema operativo gere recursos como processador, memória, armazenamento e dispositivos, permitindo também a execução de programas."
    ),
]


def generate_dynamic_chat():
    category = random.randint(0, 4)

    if category == 0:
        concept, definition = random.choice(list(concepts.items()))

        questions = [
            f"O que é {concept}?",
            f"Podes explicar {concept}?",
            f"Explica de forma simples o que é {concept}.",
        ]

        answers = [
            f"{concept.capitalize()} é {definition}.",
            f"De forma simples, {concept} é {definition}.",
            f"Podemos definir {concept} como {definition}.",
        ]

        return random.choice(questions), random.choice(answers)

    elif category == 1:
        country, continent = random.choice(list(countries.items()))

        questions = [
            f"{country} fica em que continente?",
            f"Em que continente fica {country}?",
            f"Onde fica {country}?",
        ]

        answers = [
            f"{country} fica em {continent}.",
            f"O país {country} encontra-se no continente {continent}.",
            f"Geograficamente, {country} pertence a {continent}.",
        ]

        return random.choice(questions), random.choice(answers)

    elif category == 2:
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        result = a + b

        questions = [
            f"Quanto é {a} + {b}?",
            f"Calcula {a} + {b}.",
            f"Qual é o resultado de {a} mais {b}?",
        ]

        answers = [
            f"{a} + {b} = {result}.",
            f"O resultado é {result}.",
            f"A soma de {a} e {b} é {result}.",
        ]

        return random.choice(questions), random.choice(answers)

    elif category == 3:
        subject = random.choice(subjects)
        action = random.choice(actions)

        question = (
            f"Como pode {subject} {action}?"
        )

        answer = (
            f"Pode começar por definir claramente o objetivo, identificar os recursos necessários "
            f"e dividir a tarefa em passos menores."
        )

        return question, answer

    else:
        name = random.choice([
            "Marta", "Pedro", "Ana", "Rui",
            "Sofia", "Miguel", "Joana"
        ])

        place = random.choice([
            "casa", "biblioteca", "escritório",
            "escola", "universidade"
        ])

        question = (
            f"{name} perdeu um objeto em {place}. "
            f"Qual é uma forma simples de o procurar?"
        )

        answer = (
            f"Pode começar pelos locais onde utilizou ou guardou o objeto mais recentemente "
            f"e depois alargar a procura de forma organizada."
        )

        return question, answer


def generate_chat():
    path = OUTPUT_DIR / "chat_v3.txt"

    with path.open("w", encoding="utf-8") as f:

        for _ in range(CHAT_EXAMPLES):

            if random.random() < 0.25:
                question, answer = random.choice(chat_questions)
            else:
                question, answer = generate_dynamic_chat()

            f.write(
                f"Utilizador: {question}\n"
                f"Assistente: {answer} [EOS]\n\n"
            )

    print(f"Chat V3: {CHAT_EXAMPLES:,} conversas")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("A gerar datasets V3...\n")

    generate_pt()
    generate_math()
    generate_chat()

    print("\nDatasets V3 terminados.")