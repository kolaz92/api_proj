import requests
import streamlit as st
import json

defIP, defPort = '84.201.152.230' , '8000'

def get_log():
    gres = requests.get(f"http://{defIP}:{defPort}/log")
    st.text(f'Log:\n{gres.json()}')

def main():
    st.title("Сlassification of weather types")

    image = st.file_uploader("Choose an image", type=['jpg', 'jpeg'])

    if st.button("Classify!") and image is not None:
        st.image(image)
        files = {"file": image.getvalue()}
        res = requests.post(f"http://{defIP}:{defPort}/classify", files=files).json()
        st.write(f'Class name: {res["class_name"]}, class index: {res["class_index"]}')
        get_log()

    st.title("Test of toxicity level")

    txt = st.text_input('here')

    if st.button('send'):
        dat = {'text' : txt}
        res = requests.post(f"http://{defIP}:{defPort}/clf_text", json=dat)
        st.write(f"Probability of toxicity {res.json()['prob_of_tox']}")
        get_log()

if __name__ == '__main__':
    main()
