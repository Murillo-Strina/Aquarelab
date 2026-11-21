# Aquarelab

Laboratório que adapta paletas de cores em qualquer imagem usando redes neurais convolucionais.

Envie uma foto e o Aquarelab identifica os tons principais da imagem, separa cada região pelo que ela representa e mostra a paleta resultante. Em seguida, envie uma paleta nova com o mesmo número de cores e receba a foto recolorida, com luz, sombra e textura da original preservadas.

> Status: em desenvolvimento (v1)

## O problema

Editores de cor tradicionais agrupam pixels apenas por semelhança de tom. Se uma foto tem uma camisa azul e um céu azul, os dois caem na mesma cor da paleta, e trocar uma muda a outra junto.

O Aquarelab agrupa pixels pelo que eles representam, não só pela cor que têm. Céu, pele, vegetação e roupa podem ganhar cores diferentes, mesmo quando compartilham o mesmo tom na foto original.

## Como funciona

1. O usuário envia uma foto colorida.
2. A rede identifica os tons principais da imagem, até um máximo de 8, e mapeia quais regiões pertencem a cada tom.
3. A paleta extraída é exibida.
4. O usuário envia uma paleta nova com a mesma quantidade de cores.
5. Cada cor nova é atribuída a um tom original, do mais escuro para o mais claro, preservando o contraste da imagem.
6. A foto é recolorida na hora.

## A ideia por trás do modelo

A rede não pinta a imagem. Ela entende a imagem e entrega duas saídas:

- os tons principais da foto, deixando de fora os que não são relevantes;
- um mapa indicando o quanto cada pixel pertence a cada tom.

A recoloração é feita sobre esse mapa. Em vez de substituir a cor de cada pixel pela cor nova, o Aquarelab aplica um deslocamento: se um tom azul virou vermelho, cada pixel daquela região recebe o mesmo deslocamento de cor, proporcional ao quanto pertence a ela. Assim, sombras, brilhos e texturas continuam no lugar, e a foto final parece a mesma foto pintada com outras cores.

Essa separação traz três vantagens:

- A preservação de luz e sombra é garantida pela conta, não depende da rede acertar.
- Trocar a paleta é instantâneo, porque a rede não precisa rodar de novo.
- Cada cor nova altera apenas as regiões ligadas ao tom que ela substitui.

Para separar regiões pelo significado, e não só pela cor, a rede parte de uma base pré-treinada que já reconhece objetos e texturas.

## Treinamento

O treino não usa nenhum rótulo manual. A rede aprende tentando reconstruir a foto original usando apenas a paleta e o mapa que ela mesma gerou. Ela é recompensada por reconstruções fiéis que usem poucos tons e regiões bem definidas. Qualquer base grande de fotos coloridas públicas serve como dado de treino.

## Avaliação

A rede é comparada com um método clássico de extração de paleta que agrupa pixels apenas por cor. O objetivo é mostrar, com exemplos e métricas, em quais situações a rede separa regiões que o método clássico mistura, e onde ela ainda falha.

## Roadmap v1

- [ ] Recoloração por deslocamento de cor a partir de um mapa de tons
- [ ] Atribuição de paleta por ordem de luminosidade
- [ ] Rede que extrai os tons principais e o mapa de cada região
- [ ] Treinamento por reconstrução, sem rótulos
- [ ] Interface para envio da foto e da paleta
- [ ] Comparação com o método clássico

## Próximas versões

- Rede crítica que avalia o quanto uma imagem recolorida parece natural
- Atribuição automática de cores mais sofisticada que a ordem por luminosidade
- Suporte a fotos em preto e branco

## Limitações conhecidas

A atribuição por luminosidade preserva o contraste, mas nem sempre produz a combinação que uma pessoa escolheria. Quando a paleta nova é muito diferente da original, algumas regiões podem receber cores pouco naturais, como pele em tons improváveis. A quantidade máxima de 8 tons também simplifica imagens muito complexas.

## Objetivos de aprendizado

Este projeto foi criado para aprofundar conhecimentos em redes convolucionais além da classificação de imagens, explorando redes que geram mapas como saída, uso de modelos pré-treinados como base, treinamento sem rótulos e avaliação comparada com métodos clássicos.
