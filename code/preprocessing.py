import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ast
import os
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════
# 설정
# ══════════════════════════════════════════
INPUT_PATH  = r'C:\Users\jinhu\Desktop\월 데이터마이닝\Movie\First_data\TMDB_all_movies.csv'
OUTPUT_PATH = r'C:\Users\jinhu\Desktop\월 데이터마이닝\Movie\data\tmdb_final.csv'

# data 폴더 없으면 자동 생성
os.makedirs(r'C:\Users\jinhu\Desktop\월 데이터마이닝\Movie\data', exist_ok=True)

KEEP_COLS = [
    'budget', 'revenue', 'genres', 'popularity',
    'vote_average', 'vote_count', 'runtime', 'release_date',
    'original_language', 'director', 'cast',
    'imdb_rating', 'imdb_votes', 'status'
]

# ══════════════════════════════════════════
# Step 1: 데이터 로드 및 컬럼 선택
# ══════════════════════════════════════════
print("=" * 50)
print("Step 1: 데이터 로드 및 컬럼 선택")
print("=" * 50)

df = pd.read_csv(INPUT_PATH)
print(f"원본 데이터: {df.shape[0]:,}행 · {df.shape[1]}컬럼")

df = df[KEEP_COLS]
print(f"컬럼 선택 후: {df.shape[0]:,}행 · {df.shape[1]}컬럼")

# ══════════════════════════════════════════
# Step 2: 유효 데이터 필터링
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 2: 유효 데이터 필터링")
print("=" * 50)

# Released만 유지
df = df[df['status'] == 'Released']
print(f"Released 필터링: {df.shape[0]:,}행")

# budget, revenue > $10,000
df = df[(df['budget'] > 10000) & (df['revenue'] > 10000)]
print(f"budget/revenue 필터링: {df.shape[0]:,}행")

# status 컬럼 제거
df = df.drop(columns=['status'])

# ══════════════════════════════════════════
# Step 3: 핵심 변수 결측치 제거
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 3: 핵심 변수 결측치 제거")
print("=" * 50)

before = len(df)
df = df.dropna(subset=['genres', 'release_date', 'runtime'])
print(f"결측치 제거: {before:,} → {len(df):,}행 ({before-len(df):,}개 제거)")

# ══════════════════════════════════════════
# Step 4: 이상치 처리 (IQR × 1.5)
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 4: 이상치 처리 (IQR × 1.5)")
print("=" * 50)

for col in ['budget', 'revenue', 'runtime']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"{col}: {before:,} → {len(df):,}행 ({before-len(df):,}개 제거)")
    print(f"  기준: {lower:,.0f} ~ {upper:,.0f}")

# ══════════════════════════════════════════
# Step 5: 날짜 파생 변수 생성
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 5: 날짜 파생 변수 생성")
print("=" * 50)

df['release_date'] = pd.to_datetime(df['release_date'])
df['year']  = df['release_date'].dt.year
df['month'] = df['release_date'].dt.month
df['season'] = df['month'].map({
    12: 4, 1: 4, 2: 4,   # 겨울
    3: 1,  4: 1, 5: 1,   # 봄
    6: 2,  7: 2, 8: 2,   # 여름
    9: 3, 10: 3, 11: 3   # 가을
})
df = df.drop(columns=['release_date'])
print(f"year, month, season 파생 변수 생성 완료")
print(f"season 분포:\n{df['season'].value_counts().sort_index().rename({1:'봄',2:'여름',3:'가을',4:'겨울'})}")

# ══════════════════════════════════════════
# Step 6: 장르 원-핫 인코딩
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 6: 장르 원-핫 인코딩")
print("=" * 50)

def parse_genres(g):
    try:
        items = ast.literal_eval(g)
        if isinstance(items, list):
            return [i['name'] if isinstance(i, dict) else i for i in items]
    except:
        pass
    if isinstance(g, str) and ',' in g:
        return [x.strip() for x in g.split(',')]
    return [g] if isinstance(g, str) and g else []

df['genres_list'] = df['genres'].apply(parse_genres)

# 모든 장르 추출
all_genres = set()
df['genres_list'].apply(lambda x: all_genres.update(x))
print(f"전체 장르 수: {len(all_genres)}개")
print(f"장르 목록: {sorted(all_genres)}")

# 원-핫 인코딩
for genre in sorted(all_genres):
    if genre:
        col_name = f"genre_{genre.replace(' ', '_').replace('-', '_')}"
        df[col_name] = df['genres_list'].apply(lambda x: 1 if genre in x else 0)

genre_cols = [c for c in df.columns if c.startswith('genre_')]
print(f"생성된 장르 컬럼: {len(genre_cols)}개")

df = df.drop(columns=['genres', 'genres_list'])

# ══════════════════════════════════════════
# Step 7: 감독/출연진 빈도 기반 변수화
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 7: 감독/출연진 빈도 기반 변수화")
print("=" * 50)

# 결측치 처리
df['director'] = df['director'].fillna('Unknown')
df['cast']     = df['cast'].fillna('Unknown')

# 감독 등장 횟수
director_counts = df['director'].value_counts()
TOP_N_DIRECTOR  = 50
top_directors   = set(director_counts.head(TOP_N_DIRECTOR).index)
df['director_is_top'] = df['director'].apply(lambda x: 1 if x in top_directors else 0)
print(f"상위 {TOP_N_DIRECTOR}명 감독 변수화 완료")
print(f"상위 감독 영화 비율: {df['director_is_top'].mean()*100:.1f}%")

# 출연진 (첫 번째 배우 기준)
def get_first_cast(cast_str):
    if pd.isna(cast_str) or cast_str == 'Unknown':
        return 'Unknown'
    try:
        items = ast.literal_eval(cast_str)
        if isinstance(items, list) and len(items) > 0:
            return items[0]['name'] if isinstance(items[0], dict) else items[0]
    except:
        pass
    return cast_str.split(',')[0].strip() if ',' in str(cast_str) else cast_str

df['lead_actor'] = df['cast'].apply(get_first_cast)
actor_counts     = df['lead_actor'].value_counts()
TOP_N_ACTOR      = 50
top_actors       = set(actor_counts.head(TOP_N_ACTOR).index)
df['actor_is_top'] = df['lead_actor'].apply(lambda x: 1 if x in top_actors else 0)
print(f"상위 {TOP_N_ACTOR}명 배우 변수화 완료")
print(f"상위 배우 영화 비율: {df['actor_is_top'].mean()*100:.1f}%")

df = df.drop(columns=['director', 'cast', 'lead_actor'])

# ══════════════════════════════════════════
# Step 8: 나머지 결측치 처리
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 8: 나머지 결측치 처리")
print("=" * 50)

df['imdb_rating'] = df['imdb_rating'].fillna(df['imdb_rating'].median())
df['imdb_votes']  = df['imdb_votes'].fillna(df['imdb_votes'].median())
df['original_language'] = df['original_language'].fillna('unknown')

print(f"결측치 현황:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"전체 결측치: {df.isnull().sum().sum()}개")

# ══════════════════════════════════════════
# Step 9: 파생 변수 생성 (ROI, 흥행 여부)
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 9: 파생 변수 생성")
print("=" * 50)

# ROI (투자 대비 수익률)
df['roi'] = (df['revenue'] - df['budget']) / df['budget']

# 흥행 여부 (revenue 상위 30% = 1)
threshold = df['revenue'].quantile(0.70)
df['hit'] = (df['revenue'] >= threshold).astype(int)

print(f"ROI 변수 생성 완료 (평균: {df['roi'].mean():.2f})")
print(f"흥행 기준 revenue: ${threshold:,.0f}")
print(f"흥행 비율: {df['hit'].mean()*100:.1f}% (상위 30%)")

# ══════════════════════════════════════════
# Step 10: Z-Score 정규화
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("Step 10: Z-Score 정규화")
print("=" * 50)

numeric_cols = [
    'budget', 'popularity', 'vote_average', 'vote_count',
    'runtime', 'imdb_rating', 'imdb_votes', 'year', 'roi'
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"정규화 적용 컬럼: {numeric_cols}")

# ══════════════════════════════════════════
# 최종 저장
# ══════════════════════════════════════════
print("\n" + "=" * 50)
print("최종 결과")
print("=" * 50)

df.to_csv(OUTPUT_PATH, index=False)
df_scaled.to_csv(r'C:\Users\jinhu\Desktop\월 데이터마이닝\Movie\data\tmdb_final_scaled.csv', index=False)

print(f"최종 데이터셋: {df.shape[0]:,}행 · {df.shape[1]}컬럼")
print(f"저장 완료: {OUTPUT_PATH}")
print(f"정규화 버전: C:\\Users\\jinhu\\Desktop\\월 데이터마이닝\\Movie\\data\\tmdb_final_scaled.csv")
print(f"\n컬럼 목록:")
for col in df.columns:
    print(f"  - {col}")