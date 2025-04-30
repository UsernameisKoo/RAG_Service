import streamlit as st

st.set_page_config(
    page_title="랜드마크 ALBUM",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLs5EctceaWtw8-MNhhsZzvvwvGzKu_GrYLA&s"
)

st.markdown("""
<style>
img {
    height: 250px;
    max-weight: 200px;
}
.streamlit-expanderContent div {
    display: flex;
    justify-content: center;
    font-size: 20px;
}
[data-testid="stExpanderToggleIcon"] {
    visibility: hidden;
}
.streamlit-expanderHeader {
    pointer-events: none;
}
[data-testid="StyledFullScreenButton"] {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)



st.title("streamlit 나라별 랜드마크🗼 Album")
st.markdown("**랜드마크**를 하나씩 추가해서 사진첩을 채워보세요!")

type_emoji_dict = {
    "프랑스": "💎",
    "한국": "🐯",
    "일본": "🌸",
    "중국": "🐼",
    "미국": "🗽",
    "태국": "🐘",
    "캐나다": "🍁",
    "건물": "🏯",
    "타워": "🗼",
    "동상": "🗿",
}

initial_landmarks = [
    {
        "name": "자금성",
        "types": ["중국", "건물"],
        "years": "1421",
        "image_url": "https://i.namu.wiki/i/leG_DcmwNLV2P-et1rDl-K2fSqp-FElYabZ1iwCPcsz_4RsluU6Dfso3azeqlSebdMN2kZW2_KGpVFUY67Knu_fpjMthvvmIDo-jDCnC6x2iiAfSHejoJ5vpiiuMrFvwMhdosec1cInD3tnngy4LKg.webp",
    },
    {
        "name": "자유의 여신상",
        "types": ["미국", "동상"],
        "years": "1978",
        "image_url": "https://i.namu.wiki/i/s1ovN9RrlXf7aAxVaOa8FDkOwJZ3dDsS57itn8QkIO6vZsStHA6mUUdWTQyKYMNYZ5GmC3zEUo40k3zsGAElVNXzDSllXOCHX78OoSv_IHakVaSu57bV6NyyGX4vyB_lsJIIsoFPp-fBnBHX6bj_DQ.webp",
    },
    {
        "name": "에펠탑",
        "types": ["프랑스", "타워"],
        "years": "1887",
        "image_url": "https://i.namu.wiki/i/_xkEBLw_WesEL31k7dbaXDIROw-U-sHvV9yO3vhYGXsW0qnfpuXsU3ZT1_pqGQrVPKtsQ882rKFx4zfjhZNNcnJcXS8OQhFcrFdlzmsODUR9LGvcR7EtvPpAtvinqQPWtw97R7ByKp_YxiXiL63e2g.webp",
    },
    {
        "name": "오사카성",
        "types": ["일본", "건물"],
        "years": "1583",
        "image_url": "https://i.namu.wiki/i/2TjPr4wT1_AGDn_AVcOy0WL4OdrRDcvxluclGvZrd_mze-SqUmy4-5dJ6BIuHludtTS0qc31qGYV8SOD3dLwFxkL3WqNNczGLg3QhlTBndTZ7KFqHJWj7APg80LOgXrmlOhQ5Hs4JE01RckyH8p_HQ.webp"
    },
]

example_landmark = {
    "name": "경희루",
    "types": ["한국", "건물"],
    "years": "1867",
    "image_url": "https://i.namu.wiki/i/s6GQesL_1owUAiiIh-E02u6My5JnkEw1Zu6zx6co7WP7BQb8kBMILGK4W9geuzEc_cjO4nwr3ymQYAAe28nx-etqpktY3uh_6yXNUGdR9DuMM-ks8RMpembRtRsXRRI0gfZVsn9MlI5fy00RqAB_1w.webp"
}

if "landmarks" not in st.session_state:
    st.session_state.landmarks = initial_landmarks

auto_complete = st.toggle("예시 데이터로 채우기")

DEFAULT_IMAGE = "https://cdn-icons-png.flaticon.com/512/4611/4611607.png"


with st.form(key="form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input(
            label="랜드마크 이름",
            value=example_landmark["name"] if auto_complete else ""
        )
    with col2:
        types = st.multiselect(
            label="카테고리",
            options=list(type_emoji_dict.keys()),
            max_selections=2,
            default=example_landmark["types"] if auto_complete else []
        )
    with col3:
        years = st.text_input(
            label="연도",
            value=example_landmark["years"] if auto_complete else ""
        )
    image_url = st.text_input(
        label="이미지 URL",
        value=example_landmark["image_url"] if auto_complete else ""
    )
    submit = st.form_submit_button(label="Submit")
    if submit:
        if not name:
            st.error("사진의 이름을 입력해주세요.")
        elif len(types) == 0:
            st.error("카테고리를 적어도 한개 선택해주세요.")
        elif len(years) == 0:
            st.error("사진의 연도를 입력해주세요.")
        else:
            try:
                years_int = int(years)
                if years_int < 0 or years_int > 2025:
                    st.error("연도는 0부터 2025 사이로 입력해야 합니다.")
                else:
                    st.success("사진을 추가할 수 있습니다.")
                    st.session_state.landmarks.append({
                        "name": name,
                        "types": types,
                        "years": years_int,
                        "image_url": image_url if image_url else "https://cdn-icons-png.flaticon.com/512/4611/4611607.png"
                })
            except ValueError:
                st.error("연도는 숫자로 입력해야 합니다.")


for i in range(0, len(st.session_state.landmarks), 2):
    row_landmarks = st.session_state.landmarks[i:i+2]
    cols = st.columns(2)
    for j in range(len(row_landmarks)):
        with cols[j]:
            landmark = row_landmarks[j]
            with st.expander(label=f"**{i+j+1}. {landmark['name']}**", expanded=True):
                st.image(landmark["image_url"])
                emoji_types = [f"**{type_emoji_dict[x]} {x}**" for x in landmark["types"]]
                st.markdown(" / ".join(emoji_types))
                st.markdown(f"**연도**: {landmark['years']}")              
                delete_button = st.button(label="삭제", key=i+j, use_container_width=True)
                if delete_button:
                    del st.session_state.landmarks[i+j]
                    st.rerun()

