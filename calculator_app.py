import streamlit as st
import math
from collections import Counter

# --- КОНФИГУРАЦИЯ И ДАННЫЕ ---

# Параметры LLM и нагрузки (фиксированные)
INPUT_TOKENS = 2048
OUTPUT_TOKENS_PER_REQUEST = 500
MAX_LATENCY_S = 10

# Расчетная производительность (выходных токенов/сек на 1 GPU)
# T-lite 8B (A100 40GB): 5727 t/s
# T-pro 32B (A100 80GB): 1931 t/s
MODEL_PERFORMANCE = {
    "T-lite (8B, A100 40GB)": 5727,
    "T-pro (32B, A100 80GB)": 1931,
}

GPU_TYPE_MAPPING = {
    "T-lite (8B, A100 40GB)": "A100 40GB",
    "T-pro (32B, A100 80GB)": "A100 80GB",
}

# Цены на серверы Selectel (конфигурации)
SERVER_COSTS_CONFIGS = {
    "T-lite (8B, A100 40GB)": { # Руб/мес за сервер с N GPU
        1: 100000,
        2: 180000,
        4: 280000,
        8: 500000,
    },
    "T-pro (32B, A100 80GB)": { # Руб/мес за сервер с N GPU
        1: 300000,
        2: 600000,
        4: 1200000,
        8: 2500000,
    }
}

# Цены на лицензии Compressa (цена за 1 лицензию в зависимости от общего числа GPU)
# Ключ '4' означает цена для 3-4 GPU, ключ '8' означает цена для 5+ GPU
LICENSE_PRICES_PER_GPU_TIERS = { # Руб/мес за 1 лицензию
    1: 150000,  # для 1 GPU
    2: 145000,  # для 2 GPU
    4: 140000,  # для 3-4 GPU
    8: 130000,  # для 5+ GPU
}

# --- ФУНКЦИИ РАСЧЕТА ---

def calculate_optimal_server_cost_dp(num_gpus_needed, server_prices_for_model):
    """
    Рассчитывает минимальную стоимость серверов для заданного количества GPU,
    комбинируя доступные конфигурации (1, 2, 4, 8 GPU) с помощью ДП.
    server_prices_for_model: словарь {1: цена1, 2: цена2, ...} для выбранной модели.
    """
    if num_gpus_needed <= 0:
        return 0, "Нет GPU"

    dp_cost = [float('inf')] * (num_gpus_needed + 1)
    # dp_config будет хранить список серверов (их GPU-емкость), из которых собрана конфигурация
    dp_config_breakdown = [[] for _ in range(num_gpus_needed + 1)]
    dp_cost[0] = 0

    # Доступные размеры серверных конфигураций (например, [1, 2, 4, 8])
    server_options = sorted(server_prices_for_model.keys())

    for i in range(1, num_gpus_needed + 1):
        for gpu_config_size in server_options:
            if i >= gpu_config_size:
                cost_of_this_server_config = server_prices_for_model[gpu_config_size]
                # Если предыдущее состояние (i - gpu_config_size) достижимо
                if dp_cost[i - gpu_config_size] != float('inf'):
                    current_total_cost = dp_cost[i - gpu_config_size] + cost_of_this_server_config
                    if current_total_cost < dp_cost[i]:
                        dp_cost[i] = current_total_cost
                        dp_config_breakdown[i] = dp_config_breakdown[i - gpu_config_size] + [gpu_config_size]
    
    final_cost = dp_cost[num_gpus_needed]
    if final_cost == float('inf'): # Если не удалось собрать точное количество
        # Попробуем найти ближайшее большее N', для которого есть решение,
        # но для данного ТЗ это не требуется, мы всегда можем собрать точное N из единичек.
        # Если сервер на 1 GPU всегда доступен, то float('inf') не должно быть.
        return float('inf'), "Невозможно собрать требуемую конфигурацию (ошибка в логике или ценах)"

    # Формируем читаемое описание конфигурации
    config_counts = Counter(dp_config_breakdown[num_gpus_needed])
    config_details_str_list = []
    for gpu_size, count in sorted(config_counts.items(), reverse=True): # Сначала большие серверы
        if count > 0:
            config_details_str_list.append(f"{count} x сервер(а) на {gpu_size} GPU")
    
    final_config_description = ", ".join(config_details_str_list)
    if not final_config_description and num_gpus_needed > 0 : # если что-то пошло не так
        final_config_description = "Ошибка при определении конфигурации"
    elif num_gpus_needed == 0:
         final_config_description = "Нет GPU"

    return final_cost, final_config_description

def get_compressa_license_cost(num_gpus_exact, license_tier_prices):
    if num_gpus_exact <= 0:
        return 0
    
    price_per_license = 0
    if num_gpus_exact == 1:
        price_per_license = license_tier_prices.get(1, license_tier_prices[8]) # fallback на цену для 5+
    elif num_gpus_exact == 2:
        price_per_license = license_tier_prices.get(2, license_tier_prices[8])
    elif 3 <= num_gpus_exact <= 4:
        price_per_license = license_tier_prices.get(4, license_tier_prices[8])
    elif num_gpus_exact >= 5:
        price_per_license = license_tier_prices.get(8, license_tier_prices[8]) # Цена для 8 (5+) GPU
    
    return price_per_license * num_gpus_exact

# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(layout="wide")
st.title("Калькулятор стоимости обеспечения нагрузки на LLM")
st.markdown("Рассчитайте примерную ежемесячную стоимость инфраструктуры и лицензий Compressa.")

st.sidebar.header("Параметры для расчета")

model_choice = st.sidebar.selectbox(
    "1. Выберите модель LLM:",
    list(MODEL_PERFORMANCE.keys())
)

num_users = st.sidebar.slider(
    "2. Количество одновременных пользователей:",
    min_value=5, max_value=500, value=50, step=1
)

# --- ЛОГИКА РАСЧЕТА НА ОСНОВЕ ВВОДА ---
gpu_output_tps_per_gpu = MODEL_PERFORMANCE[model_choice]
gpu_type_for_model = GPU_TYPE_MAPPING[model_choice]
server_prices_for_model = SERVER_COSTS_CONFIGS[model_choice]

# RPS (requests per second) per GPU
rps_per_gpu = gpu_output_tps_per_gpu / OUTPUT_TOKENS_PER_REQUEST

# Total RPS needed for all users with desired latency
total_rps_needed = num_users / MAX_LATENCY_S

# Number of GPUs needed (exact, float then ceiling)
if rps_per_gpu > 0:
    num_gpus_needed_exact = math.ceil(total_rps_needed / rps_per_gpu)
else:
    num_gpus_needed_exact = 0 # Should not happen with valid performance data

if num_gpus_needed_exact == 0 and total_rps_needed > 0: # Если нагрузка очень мала, но есть
    num_gpus_needed_exact = 1
elif total_rps_needed == 0: # Если 0 пользователей
    num_gpus_needed_exact = 0


# Calculate server costs
server_cost, server_config_details = calculate_optimal_server_cost_dp(num_gpus_needed_exact, server_prices_for_model)

# Calculate Compressa license costs
license_cost = get_compressa_license_cost(num_gpus_needed_exact, LICENSE_PRICES_PER_GPU_TIERS)

# Total cost
total_cost = server_cost + license_cost

# --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
st.subheader(f"Результаты расчета для {num_users} пользователей и модели {model_choice.split(' (')[0]}")

if model_choice == "T-lite (8B, A100 40GB)":
    st.info(f"ℹ️ Для модели T-lite используются GPU **{gpu_type_for_model}**.")
elif model_choice == "T-pro (32B, A100 80GB)":
    st.info(f"ℹ️ Для модели T-pro используются GPU **{gpu_type_for_model}**.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Необходимо GPU (расчетно)", value=f"{num_gpus_needed_exact} шт.")
    if num_gpus_needed_exact > 0:
        st.caption(f"Конфигурация серверов: {server_config_details}")
    else:
        st.caption("Конфигурация серверов: нет GPU")


with col2:
    st.metric(label="Стоимость серверов Selectel (в месяц)", value=f"{server_cost:,.0f} руб.")
    st.caption(f"На базе карт {gpu_type_for_model}")


with col3:
    st.metric(label="Стоимость лицензий Compressa (в месяц)", value=f"{license_cost:,.0f} руб.")
    st.caption(f"Для {num_gpus_needed_exact} GPU")

st.markdown("---")
st.header(f"ИТОГО: {total_cost:,.0f} руб. в месяц")
st.markdown("---")


with st.expander("Допущения и детали расчета (нажмите, чтобы развернуть)"):
    st.markdown(f"""
    - **Параметры запросов:**
        - Средний вход: {INPUT_TOKENS} токенов.
        - Средний выход: {OUTPUT_TOKENS_PER_REQUEST} токенов на запрос.
    - **Производительность:**
        - Максимальная допустимая задержка на ответ: {MAX_LATENCY_S} секунд.
        - Расчетная производительность 1 GPU {model_choice.split(' (')[0]} ({gpu_type_for_model}): {gpu_output_tps_per_gpu} выходных токенов/сек.
        - Это значение ({gpu_output_tps_per_gpu} т/с) используется для расчета количества запросов, которые 1 GPU может обработать в секунду с учетом длины ответа.
    - **Расчет необходимого количества GPU:**
        - Общая требуемая пропускная способность (запросов/сек) = Количество пользователей / {MAX_LATENCY_S} сек.
        - Количество GPU = Округление вверх (Общая требуемая пропускная способность / Запросов/сек на 1 GPU).
    - **Стоимость серверов (Selectel):**
        - Рассчитывается путем оптимального подбора комбинаций серверов (на 1, 2, 4, или 8 GPU), чтобы обеспечить необходимое количество GPU с минимальными затратами.
        - Цены на серверы зависят от выбранной модели (типа GPU).
    - **Стоимость лицензий Compressa:**
        - Цена за одну лицензию зависит от общего количества используемых GPU (предусмотрены скидки за объем). 1 GPU = 1 лицензия.
    - **Важно:**
        - Все расчеты являются оценочными. Реальная производительность может отличаться.
        - Цены на серверы и лицензии актуальны на момент создания калькулятора и могут изменяться.
        - Данный калькулятор не учитывает другие возможные затраты (сетевой трафик, хранилище и т.д.).
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Разработано для демонстрационных целей.")
