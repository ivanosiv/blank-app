import streamlit as st
import dotenv
import os
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai
import random

dotenv.load_dotenv()

# Funções de conversão de imagem para base64 e vice-versa
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

# Função para converter mensagens para o formato Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }
        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)
        prev_role = message["role"]
    return gemini_messages

# Função para consultar e transmitir a resposta do Gemini
def stream_llm_response(model_params, api_key=None, prompt_override=None):
    response_message = ""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_params["model"],
        generation_config={
            "temperature": model_params["temperature"],
        }
    )
    
    # Utiliza o prompt_override se fornecido, caso contrário, utiliza mensagens do session_state
    if prompt_override:
        gemini_messages = [{"role": "user", "parts": [prompt_override]}]
    else:
        gemini_messages = messages_to_gemini(st.session_state.messages)
    
    for chunk in model.generate_content(contents=gemini_messages, stream=True):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text
    
    # Adicionar a resposta do assistente ao histórico de mensagens
    st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_message}]})

# Função para análise de imagem de prato
def analyze_dish_image(image, google_api_key, idade, peso, altura, imc, nivel_atividade):
    # Criar o prompt oculto para a análise calórica
    prompt = (
        f"Atue como um nutricionista. Aqui estão algumas informações para ajudar a estimar as calorias do prato: "
        f"idade {idade}, peso {peso} kg, altura {altura} m, IMC {imc} e nível de atividade física {nivel_atividade}. "
        "Por favor, forneça uma estimativa calórica para este prato com base nas informações fornecidas."
    )

    # Adicionar apenas a imagem ao histórico de mensagens sem o prompt de texto
    st.session_state.messages.append({
        "role": "user", 
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image)}"}}]
    })

    # Enviar o prompt oculto para o modelo e exibir a resposta
    with st.chat_message("assistant"):
        st.write_stream(stream_llm_response({"model": "gemini-1.5-flash", "temperature": 0.3}, google_api_key, prompt_override=prompt))

# Função para recomendar receitas com base nos ingredientes
def recommend_recipes_with_ingredients(image, google_api_key):
    restricoes = ", ".join(st.session_state.restricoes_alimentares)
    st.session_state.messages.append({
        "role": "user", 
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image)}"}}]
    })
    st.session_state.messages.append({
        "role": "user", 
        "content": [{"type": "text", "text": f"Baseando-se nos ingredientes da imagem e nas seguintes restrições alimentares: {restricoes}, recomende receitas saudáveis somente com os ingredientes da imagem para o perfil do usuário."}]
    })
    with st.chat_message("assistant"):
        st.write_stream(stream_llm_response({"model": "gemini-1.5-flash", "temperature": 0.3}, google_api_key))

def main():
    st.set_page_config(page_title="App Nutrição", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
    st.title("App Nutrição 💬")

    # Inicializar session_state para mensagens e restrições, se ainda não existirem
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "restricoes_alimentares" not in st.session_state:
        st.session_state.restricoes_alimentares = []
    if "prompt_enviado" not in st.session_state:
        st.session_state.prompt_enviado = False

    # Barra lateral para opções e configurações
    with st.sidebar:
        google_api_key = st.text_input("Introduza sua chave API do Google", value=os.getenv("GOOGLE_API_KEY") or "", type="password")
        
        st.divider()
        st.write("### **Dados de Saúde do Usuário**")
        idade = st.number_input("Idade", min_value=1, max_value=120, step=1)
        peso = st.number_input("Peso (kg)", min_value=1.0)
        altura = st.number_input("Altura (m)", min_value=0.5)
        imc = round(peso / (altura ** 2), 2) if altura else 0
        nivel_atividade = st.selectbox("Nível de Atividade Física", ["Sedentário", "Moderado", "Ativo", "Muito Ativo"])
        restricoes_alimentares = st.multiselect("Restrições Alimentares", ["Diabetes", "Hipertensão", "Alergias Alimentares", "Doenças Celíacas", "Vegetariano", "Vegano", "Low Carb", "Keto"])
        st.session_state.restricoes_alimentares = restricoes_alimentares
        st.write(f"**IMC Calculado:** {imc}")
        
        st.divider()
        st.write("### **Escolha uma Opção de Análise**")
        uploaded_image = st.file_uploader("Carregar uma imagem de refeição ou ingredientes:", type=["png", "jpg", "jpeg"])
        option = st.selectbox("Escolha a análise desejada", ["Calcular Calorias do Prato", "Recomendar Receitas com Ingredientes"])

        # Botão para resetar a conversa
        if st.button("🗑️ Resetar conversa"):
            st.session_state.messages.clear()
            st.session_state.prompt_enviado = False

    # Área principal para o chat e execução da análise selecionada
    st.subheader("Conversa")

    # Executa a análise selecionada quando uma imagem é carregada
    if uploaded_image:
        image = Image.open(uploaded_image)
        if option == "Calcular Calorias do Prato":
            analyze_dish_image(image, google_api_key, idade, peso, altura, imc, nivel_atividade)
        elif option == "Recomendar Receitas com Ingredientes":
            recommend_recipes_with_ingredients(image, google_api_key)

    # Mostra o histórico de mensagens na área principal
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                elif content["type"] == "image_url":      
                    st.image(content["image_url"]["url"])

    # Input de chat para mensagens contínuas
    if prompt := st.chat_input("Digite uma pergunta ou pedido de recomendação..."):
        st.session_state.messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": prompt}]
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.write_stream(stream_llm_response({"model": "gemini-1.5-flash", "temperature": 0.3}, google_api_key))

if __name__ == "__main__":
    main()
