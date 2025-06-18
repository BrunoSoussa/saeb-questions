# SAEB Questions API

API para processamento de respostas de questões do SAEB (Sistema de Avaliação da Educação Básica).

## Estrutura da Resposta

A API retorna um objeto JSON com a seguinte estrutura:

```json
{
    "blocks": [
        {
            "block": number,
            "response": {
                "is_valid_img": boolean,
                "questions_marked_processed": [
                    {
                        "answer": string,
                        "question": number
                    }
                ]
            }
        }
    ],
    "status": string
}
```

### Campos

- `blocks`: Array contendo os blocos de questões processados
  - `block`: Número identificador do bloco (1, 2, etc.)
  - `response`: Objeto contendo as respostas do bloco
    - `is_valid_img`: Indica se a imagem do bloco é válida
    - `questions_marked_processed`: Array com as questões marcadas
      - `answer`: Letra da alternativa marcada (a, b, c, d, etc.)
      - `question`: Número da questão
- `status`: Status da requisição ("success" em caso de sucesso)

## Exemplo de Saída

```json
{
    "blocks": [
        {
            "block": 1,
            "response": {
                "is_valid_img": true,
                "questions_marked_processed": [
                    {
                        "answer": "a",
                        "question": 1
                    },
                    {
                        "answer": "b",
                        "question": 2
                    },
                    {
                        "answer": "c",
                        "question": 3
                    }
                ]
            }
        },
        {
            "block": 2,
            "response": {
                "is_valid_img": true,
                "questions_marked_processed": [
                    {
                        "answer": "b",
                        "question": 1
                    },
                    {
                        "answer": "d",
                        "question": 2
                    },
                    {
                        "answer": "a",
                        "question": 3
                    }
                ]
            }
        }
    ],
    "status": "success"
}
```

## Como Usar

1. Envie uma requisição para o endpoint da API com a imagem do gabarito
2. A API processará a imagem e retornará as respostas no formato especificado
3. Verifique o campo `status` para confirmar o sucesso da operação
4. Acesse o array `blocks` para obter as respostas de cada bloco

## Requisitos

- Imagem do gabarito em formato legível
- Cada bloco de questões deve estar claramente identificado
- As alternativas devem estar marcadas de forma clara e legível

## Status de Resposta

- `status: "success"`: A requisição foi processada com sucesso
- `status: "error"`: Ocorreu um erro ao processar a requisição (detalhes no campo `message`)
