## README

# Previsão de peça a ser trocada a partir do problema descrito no chamado

## Descrição

Esta aplicação permite prever a peça a ser trocada a partir do problema descrito em um chamado de suporte técnico com base em dados fornecidos pelo usuário, utilizando um modelo de aprendizado de máquina treinado com BERT para processar descrições textuais. O sistema também fornece uma lista de exemplos de entradas para orientar o usuário.

## Funcionalidades

- Previsão da peça a ser trocada em um chamado com base nos dados inseridos, modelo, problema e tipo de equipamento.
- Exibição de uma lista de exemplos de chamados para referência.
- Possibilidade de carregar mais exemplos aleatoriamente.

## Requisitos

- Python 3.6 ou superior
- Bibliotecas Python:
  - Flask
  - pandas
  - numpy
  - torch
  - transformers
  - keras
  - joblib

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual:

   - No Windows:

     ```bash
     venv\Scripts\activate
     ```

   - No Linux/Mac:

     ```bash
     source venv/bin/activate
     ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Certifique-se de que os arquivos necessários estão no caminho correto:

   - Modelo BERT e tokenizer (`bert-base-uncased`)
   - Arquivos de scaler e encoder (`scaler15.pkl` e `label_encoder15.pkl`)
   - CSV com mapeamento de códigos (`cod_solic3.csv`)
   - CSV com exemplos (`amostra_exemplo.csv`)
   - Modelo Keras (`model15.h5`)

## Uso

1. Execute a aplicação Flask:

   ```bash
   python app.py
   ```

2. Abra seu navegador e acesse `http://127.0.0.1:5000`.

3. Preencha os campos do formulário com os dados do chamado e clique em "Enviar".

4. Veja a previsão da categoria do chamado e as probabilidades associadas.

5. Use a seção de exemplos à direita para ver exemplos de entradas de chamados.

6. Clique em "Mais Exemplos" para carregar mais exemplos aleatórios.

## Estrutura do Projeto

```
.
├── app.py                  # Código principal da aplicação Flask
├── requirements.txt        # Dependências do projeto
├── templates
│   └── index.html          # Template HTML principal
└── static
    └── styles.css          # Arquivo de estilos CSS
    └── amostra_exemplo.csv # Arquivo CSV com exemplos
```

## Contribuição

1. Faça um fork do projeto.
2. Crie uma nova branch (`git checkout -b feature/nova-feature`).
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`).
4. Faça um push para a branch (`git push origin feature/nova-feature`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato

Se você tiver dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou entrar em contato com o autor do projeto.