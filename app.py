import streamlit as st
import pandas as pd
import json
import os
import re
import datetime
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv
from langfuse.decorators import observe
from langfuse.openai import OpenAI
import boto3
from pycaret.regression import load_model, predict_model

from utils import utils_css

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize S3 client
s3 = boto3.client("s3")

@st.cache_data
def get_model():
    try:
        return load_model("time_sec_model", platform="aws", authentication={"bucket": "wk1", "path": "zadanie_9/models"})
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {str(e)}.")
        raise Exception("Nie znaleziono modelu.")

def html(st, html):
    st.markdown(html, unsafe_allow_html=True)

@dataclass
class RunnerInfo:
    age: int
    sex: str
    time_5k: float  # time in seconds
    pace_5k: str # pace as string

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gender": self.sex.upper(),  # Match the model's expected format
            "age": self.age,
            "5_km_sec": self.time_5k
        }

    def sex_str(self) -> str:
        if self.sex == 'F':
            return 'kobieta'
        elif self.sex == 'M':
            return 'mężczyzna'
        else:
            return 'nie podano'

    def time_5k_str(self) -> str:
        """Format seconds to MM:SS"""
        minutes = int(self.time_5k // 60)
        remaining_seconds = int(self.time_5k % 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def pace_5k_str(self) -> str:
        return self.pace_5k if not self.pace_5k is None else 'nie podano'


def parse_time(time_str: str) -> float:
    """Parse time string in format MM:SS to seconds"""
    match = re.match(r"^(\d+)[:](\d{2})$", time_str)
    if not match:
        match = re.match(r"^(\d+)[\"\'](\d{2})[\"\']*$", time_str)
    if match:
        minutes, seconds = map(int, match.groups())
        return minutes * 60 + seconds
    raise ValueError(f"Could not parse time: {time_str}")

@observe()
def parse_user_input(text: str) -> RunnerInfo:
    """Parse user input using OpenAI to extract runner information"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract information about the runner from the text."
                        "Return a JSON object with the following fields:"
                            "age (int - age),"
                            "sex (string 'M' for male or 'F' for female),"
                            "time_5k (string in MM:SS format if time given),"
                            "pace_5k (string in MM:SS format if pace given)",
                },
                {
                    "role": "user",
                    "content": text
                },
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)

        time_5k = data.get("time_5k")
        pace_5k = data.get("pace_5k")
        if time_5k is None and pace_5k is None:
            raise ValueError("Nie podano czasu na 5 km.")
        # Parse time if it's in MM:SS format
        if time_5k:
            time_5k = parse_time(time_5k)
        else:
            time_5k = parse_time(pace_5k) * 5

        if time_5k < 15 + 60 or time_5k > 120 * 60:
            raise ValueError("Podany czas na 5 km przekracza ludzkie pojęcie.")

        sex = data.get("sex")
        if sex is None:
            raise ValueError("Nie podano płci.")
        else:
            if sex not in ["F", "M"]:
                raise ValueError("Spróbuj bardziej wyraźnie określić swoją płeć.")

        age = data.get("age")
        if age is None:
            raise ValueError("Nie podano wieku.")
        else:
            if age < 18 or age > 105:
                raise ValueError("W tym wieku nie możesz startować w maratonie.")

        runner = RunnerInfo(age=age, sex=sex, time_5k=time_5k, pace_5k=pace_5k)

        return runner
    except ValueError as e:
        st.error(f"Zły format danych: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
        raise

def main():
    if "user_input" not in st.session_state:
        st.session_state.user_input = "Jestem 28-letnią kobietą, a moje średnie tempo na 5 km to 4\"17'."
    if "runner_info" not in st.session_state:
        st.session_state.runner_info = None
    if "page" not in st.session_state:
        st.session_state.page = "input"
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    st.set_page_config(
        page_title="Oszacuj swoje miejsce w półmaratonie",
        page_icon="🏃",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    html(st, utils_css.better_styling_css())

    if st.session_state.page == "input":
        display_input_page()
    elif st.session_state.page == "data":
        display_data_page()
    elif st.session_state.page == "results":
        display_results_page()

def display_input_page():
    html(st, '<h1 class="main-header">🏃 Prognoza czasu w Półmaratonie Wrocławskim</h1>')
    tab1, tab2 = st.tabs(['Prognoza', 'Tempo/Prędkość'])
    with tab1:
        html(st, '<h3 class="sub-header">💬 Już za chwilę dowiesz się jaki czas uzyskałbyś w Półmaratonie Wrocławskim.</h3>')
        html(st, """
            <div class="highlight-box">
                <p>Powiedz coś o sobie, ile masz lat, czy jesteś mężczyzną czy kobietą, jaki masz czas (lub jakie jest Twoje tempo) na 5 km.<br>
                W razie wątpliwości skorzystaj z tabeli Tempo/Prędkość.<br>
                Poniżej przykładowe odpowiedzi:</p>
                <ul>
                    <li>Jestem 35-letnim mężczyzną, a mój czas na 5 km to 22:30.</li>
                    <li>Mam 28 lat, jestem kobietą a moje średnie tempo na 5 km to 4"59'</li>
                </ul>
            </div>
        """)

        user_input = st.text_area(
            "Opisz siebie i swój czas lub tempo na 5 km",
            st.session_state.user_input,
            height=120,
            label_visibility="collapsed",
        )

        analyze_button = st.button("🔍 Dalej", key="analyze_btn", use_container_width=True)
    with tab2:
        html(st, """
            <table>
                <tr>
                    <th>Tempo min"sec'/km</th><th>Prędkość km/h</th><th></th>
                </tr>
                <tr>
                    <td>3"20'</td><td>18</td><td></td>
                </tr>
                <tr>
                    <td>3"31'</td><td>17</td><td>tak biegają najlepsi</td>
                </tr>
                <tr>
                    <td>3"45'</td><td>16</td><td></td>
                </tr>
                <tr>
                    <td>4"00'</td><td>15</td><td></td>
                </tr>
                <tr>
                    <td>4"17'</td><td>14</td><td></td>
                </tr>
                <tr>
                    <td>4"47'</td><td>13</td><td></td>
                </tr>
                <tr>
                    <td>5"00'</td><td>12</td><td></td>
                </tr>
                <tr>
                    <td>5"27'</td><td>11</td><td></td>
                </tr>
                <tr>
                    <td>6"00'</td><td>10</td><td></td>
                </tr>
                <tr>
                    <td>6"40'</td><td>9</td><td></td>
                </tr>
                <tr>
                    <td>7"30'</td><td>8</td><td></td>
                </tr>
                <tr>
                    <td>8"34'</td><td>7</td><td></td>
                </tr>
                <tr>
                    <td>10"00'</td><td>6</td><td></td>
                </tr>
                <tr>
                    <td>12"00'</td><td>5</td><td></td>
                </tr>
                <tr>
                    <td>15"00'</td><td>4</td><td>tempo marszu</td>
                </tr>
            </table>
        """)

    if analyze_button:
        try:
            st.session_state.runner_info = parse_user_input(user_input)
            st.session_state.user_input = user_input
            st.session_state.page = "data"
            st.rerun()
        except Exception as e:
            html(st, f'<div class="warning-box">❌ Nie udało się przeanalizować danych: {str(e)}</div>')

def display_data_page():

    with st.spinner("Analizowanie danych..."):
        runner = st.session_state.runner_info
        html(st, f"""
            <div class="success-box">
                <h6 style="margin-top: 0;">✅ ... i co my tu mamy...</h6>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 0.5rem; font-weight: bold;">👤 Wiek:</td>
                        <td style="padding: 0.5rem;">{runner.age} lat</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; font-weight: bold;">⚧️ Płeć:</td>
                        <td style="padding: 0.5rem;">{runner.sex_str()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; font-weight: bold;">⏱️ Czas na 5 km:</td>
                        <td style="padding: 0.5rem;">{runner.time_5k_str()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; font-weight: bold;">⏱️ Tempo przez 5 km:</td>
                        <td style="padding: 0.5rem;">{runner.pace_5k_str()}</td>
                    </tr>
                </table>
            </div>
        """)

    estimate_button = st.button("🚀 Oszacuj czas w półmaratonie", key="estimate_btn", use_container_width=True)

    if estimate_button:
        with st.spinner("Obliczanie czasu..."):
            try:
                model = get_model()

                input_data = pd.DataFrame([st.session_state.runner_info.to_dict()])

                # ML model prediction
                pred = predict_model(model, data=input_data)
                prediction_seconds = int(pred["prediction_label"].iloc[0])

                st.session_state.prediction_results = {
                    "seconds": prediction_seconds,
                    "formatted": datetime.timedelta(seconds=prediction_seconds)
                }

                st.session_state.page = "results"
                st.rerun()

            except Exception as e:
                st.error(f"Nie udało się oszacować czasu półmaratonu: {str(e)}")

    back_button()


def display_results_page():

    html(st, '<h1 class="main-header">🏆 Wyniki Predykcji Czasu Półmaratonu</h1>')

    if st.session_state.prediction_results is None:
        html(st, '<div class="warning-box">❌ Brak wyników predykcji. Wróć do strony głównej.</div>')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Wróć do strony głównej", use_container_width=True):
                st.session_state.page = "input"
                st.rerun()
        return

    results = st.session_state.prediction_results

    runner = st.session_state.runner_info

    html(st, '<h2 class="sub-header">👤 Dane biegacza</h2>')

    col1, col2, col3 = st.columns(3)
    with col1:
        html(st, f"""
            <div class="metric-container">
                <div class="metric-label">👤 Wiek</div>
                <div class="metric-value">{runner.age} lat</div>
            </div>
        """)
    with col2:
        html(st, f"""
            <div class="metric-container">
                <div class="metric-label">⚧️ Płeć</div>
                <div class="metric-value">{runner.sex_str()}</div>
            </div>
        """)
    with col3:
        html(st, f"""
            <div class="metric-container">
                <div class="metric-label">⏱️ Czas 5 km</div>
                <div class="metric-value">{runner.time_5k_str()}</div>
            </div>
        """)
    html(st, "</div>")
    html(st, '<div class="divider"></div>')
    html(st, '<h2 class="sub-header">🔮 Wyniki predykcji</h2>')
    html(st, f"""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: 700;">{results['formatted']}</div>
        </div>
    """)
    html(st, "</div>")

    back_button()

def back_button():
    html(st, '<div class="divider"></div>')
    if st.button("🔄 Wróć do strony głównej", use_container_width=True):
        st.session_state.runner_info = None
        st.session_state.prediction_results = None
        st.session_state.page = "input"
        st.rerun()


if __name__ == "__main__":
    main()
