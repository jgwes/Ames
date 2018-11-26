import pandas as pd
from pathlib import Path

# save filepath to variable for easier access
m_file_path = Path("C:/Users/johnw_000/Desktop/kaggle/learning/")
m_file = m_file_path / "melb_data.csv"
m_data = pd.read_csv(m_file)
print(m_data.describe())
