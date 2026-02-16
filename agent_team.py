
import os
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# ---------------------------------------------------------
# 1. 모델(두뇌) 연결 설정
# - CrewAI 최신 버전에서는 LLM 클래스를 사용하여 Ollama를 직접 지정합니다.
# - 형식: "ollama/모델명"
# ---------------------------------------------------------

# PM & Senior Dev용 (EXAONE 3.5)
exaone = LLM(model="ollama/exaone3.5", base_url="http://localhost:11434")

# Code Reviewer용 (Llama 3)
llama3 = LLM(model="ollama/llama3", base_url="http://localhost:11434")

# Tester용 (Phi-3)
phi3 = LLM(model="ollama/phi3", base_url="http://localhost:11434")


# ---------------------------------------------------------
# 2. 에이전트(직원) 채용 및 역할 부여
# ---------------------------------------------------------

# [PM]: 프로젝트 매니저
project_manager = Agent(
    role='Project Manager (PM)',
    goal='프로젝트의 요구사항을 명확히 정의하고 개발 방향을 한국어로 지시',
    backstory='''당신은 경험이 풍부한 프로젝트 매니저입니다. 
    사용자의 모호한 아이디어를 구체적인 기능 명세서로 변환합니다.
    팀원들에게 명확한 업무를 지시하며, 항상 한국어로 소통합니다.''',
    llm=exaone,
    verbose=True
)

# [Senior Dev]: 수석 개발자
senior_developer = Agent(
    role='Senior Python Developer',
    goal='PM의 기획서에 따라 고품질의 Python 코드를 완벽하게 작성',
    backstory='''당신은 최고의 Python 개발자입니다. 
    복잡한 로직을 수학적 사고로 해결하며, 클린 코드(Clean Code) 원칙을 준수합니다.
    주석을 잘 달아 코드의 이해를 돕습니다.''',
    llm=exaone,
    verbose=True
)

# [Reviewer]: 코드 리뷰어
code_reviewer = Agent(
    role='Code Reviewer',
    goal='작성된 코드의 버그를 찾고 보안 취약점이나 비효율적인 부분을 지적',
    backstory='''당신은 매우 꼼꼼하고 직설적인 코드 리뷰어입니다.
    Llama 3의 냉철한 시각으로 코드를 분석하며, 잠재적인 오류를 찾아냅니다.
    문제가 없다면 "완벽합니다"라고 칭찬하지만, 보통은 개선점을 찾아냅니다.''',
    llm=llama3,
    verbose=True
)

# [Tester/Writer]: QA 및 문서화
tester = Agent(
    role='Quality Assurance Engineer',
    goal='코드를 실행하기 위한 방법(README)과 테스트 케이스 작성',
    backstory='''당신은 사용자가 코드를 쉽게 실행할 수 있도록 돕는 QA 엔지니어입니다.
    복잡한 설명보다 따라하기 쉬운 단계별 가이드를 작성하는 것을 좋아합니다.''',
    llm=phi3,
    verbose=True
)


# ---------------------------------------------------------
# 3. 작업(Task) 정의
# ---------------------------------------------------------

# ---------------------------------------------------------
# 파일 기반 I/O 설정 (team_talk.md)
# ---------------------------------------------------------
talk_file = os.path.join("picgo", "team_talk.md")

# 1. 파일에서 주제 읽기
if not os.path.exists(talk_file):
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(talk_file), exist_ok=True)
    with open(talk_file, "w", encoding="utf-8") as f:
        f.write("# Project Topic\nPython으로 만드는 이미지 생성 앱\n계획 작성해줘") # 기본값
    print(f"[{talk_file}] 파일이 없어 새로 생성했습니다. 주제를 수정 후 다시 실행하세요.")

with open(talk_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
    # 구분선("=") 혹은 결과 헤더("### [Agent Team Result]") 이전 내용만 유효한 입력으로 간주
    valid_lines = []
    for line in lines:
        if line.startswith("=") or line.startswith("### [Agent Team Result]"):
            break
        valid_lines.append(line)
        
    # 빈 줄 제외하고 하나의 문자열로 합침 (여러 줄 입력 지원)
    user_input_lines = [line.strip() for line in valid_lines if line.strip()]
    
    if user_input_lines:
        user_input = " ".join(user_input_lines) # 줄바꿈 대신 공백으로 연결하거나 "\n".join() 사용 가능
    else:
        user_input = "Python으로 만드는 간단한 계산기" # 내용이 없을 경우 기본값

print(f"## 읽어온 주제: {user_input} ##")
user_topic = user_input # 전체 입력을 그대로 주제로 사용

task1_plan = Task(
    description=f'사용자가 요청한 "{user_topic}"에 대한 기능 명세서와 개발 계획을 한국어로 상세히 작성하세요.',
    expected_output='기능 기능을 포함한 마크다운 형식의 개발 계획서',
    agent=project_manager
)

task2_code = Task(
    description=f'PM의 계획서를 바탕으로 "{user_topic}"의 전체 Python 소스 코드를 작성하세요. 하나의 파일로 실행 가능해야 합니다.',
    expected_output='완벽하게 동작하는 Python 소스 코드',
    agent=senior_developer
)

task3_review = Task(
    description='작성된 코드를 면밀히 리뷰하여 버그나 개선할 점을 찾으세요. 심각한 문제가 있다면 수정된 코드를 제안하세요.',
    expected_output='코드 리뷰 결과 보고서 (버그 유무 및 수정 제안)',
    agent=code_reviewer
)

task4_doc = Task(
    description='최종 코드의 실행 방법(README)과 테스트 시나리오를 작성하세요.',
    expected_output='README.md 내용 및 테스트 케이스 목록',
    agent=tester
)


# ---------------------------------------------------------
# 4. 팀 결성 및 프로젝트 시작
# ---------------------------------------------------------
dev_team = Crew(
    agents=[project_manager, senior_developer, code_reviewer, tester],
    tasks=[task1_plan, task2_code, task3_review, task4_doc],
    process=Process.sequential,
    verbose=True
)

print(f"### Antigravity Agent Team: '{user_topic}' 프로젝트 시작 ###")
result = dev_team.kickoff()

print("\n\n################################################")
print("## 최종 결과물 ##")
print(result)

# ---------------------------------------------------------
# 5. 결과 저장 및 파일 기록
# ---------------------------------------------------------

# 5-1. 각 단계별 결과물 저장 (중요: 코드 유실 방지)
output_dir = "picgo"
os.makedirs(output_dir, exist_ok=True)

# Helper function to safely get output string
def get_task_output(task):
    if hasattr(task, 'output') and task.output:
        return task.output.raw if hasattr(task.output, 'raw') else str(task.output)
    return "No output generated."

# Save Plan
with open(os.path.join(output_dir, "picgo_plan.md"), "w", encoding="utf-8") as f:
    f.write(get_task_output(task1_plan))

# Save Code (Raw)
with open(os.path.join(output_dir, "picgo_local_raw.py"), "w", encoding="utf-8") as f:
    f.write(get_task_output(task2_code))

# Save Review
with open(os.path.join(output_dir, "picgo_review.md"), "w", encoding="utf-8") as f:
    f.write(get_task_output(task3_review))

# Save README/Doc
with open(os.path.join(output_dir, "picgo_readme.md"), "w", encoding="utf-8") as f:
    f.write(get_task_output(task4_doc))


# 5-2. 통합 로그(team_talk.md)에 결과 요약 기록
# 전체 결과를 다 넣으면 너무 길어지므로, 최종 결과(result)와 파일 저장 위치만 기록
with open(talk_file, "a", encoding="utf-8") as f:
    f.write("\n\n" + "="*50 + "\n")
    f.write(f"### [Agent Team Result] ({user_topic})\n\n")
    f.write(f"✅ 작업이 완료되었습니다. 결과물은 `{output_dir}` 폴더에 저장되었습니다.\n\n")
    f.write(f"- 📄 기획서: `picgo_plan.md`\n")
    f.write(f"- 💻 소스코드: `picgo_local_raw.py`\n")
    f.write(f"- 🔍 리뷰보고서: `picgo_review.md`\n")
    f.write(f"- 📝 설명서: `picgo_readme.md`\n\n")
    f.write("#### 최종 요약 (Output Summary)\n")
    f.write(str(result))
    f.write("\n" + "="*50 + "\n")

print(f"\n✅ 모든 작업 결과가 '{output_dir}' 폴더에 개별 파일로 저장되었습니다.")
