'''
	1.	import os- Imports Pytho's operating system interface to access environment variables and file paths.
	2.	import asyncio- Imports Python's asynchronous I/O framework to enable concurrent execution using async/await.
	3.	import json- Imports tools for encoding and decoding JSON data used in API interactions and data serialization.
	4.	from dotenv import load_dotenv- Imports the function to load environment variables from a .env file into the process.
	5.	from typing import List- Imports the List type hint to specify lists of specific data types in class definitions.
	6.	from openai import OpenAI- Imports the OpenAI API client to send requests to OpenAI models.
	7.	from agents import Agent, FunctionTool, Runner - Imports components from OpenAI’s Agents SDK: Agent for defining AI behavior, FunctionTool for tool support, and Runner to execute conversations.
	8.	from pydantic import BaseModel, Field, ConfigDict - Imports Pydantic classes used to define and validate structured data models (e.g., user profiles, outputs).
'''

import os
import asyncio
import json
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from agents import Agent, FunctionTool, Runner
from pydantic import BaseModel, Field, ConfigDict

# 🔒 Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 🔒 Model configuration and pricing
MODEL_NAME = "gpt-4o-mini"
model_prices = {
    "gpt-4o-mini": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

# 🔒 Predefined exercises by difficulty
basic_exercises = [
    {"name": "Seated Leg Lifts", "duration": 10, "description": "Strengthens thigh and hip muscles.", "warning": "Sit in a sturdy chair. Stop if discomfort occurs."},
    {"name": "Ankle Circles", "duration": 10, "description": "Improves ankle flexibility and circulation.", "warning": "Keep foot lifted and rotate slowly."},
    {"name": "Arm Raises", "duration": 10, "description": "Strengthens shoulders and improves circulation.", "warning": "Raise arms slowly to shoulder height."},
    {"name": "Neck Rotations", "duration": 10, "description": "Reduces neck stiffness.", "warning": "Rotate head gently side to side."}
]

standard_exercises = [
    {"name": "Heel-to-Toe Walk", "duration": 15, "description": "Improves balance and gait.", "warning": "Use a wall for support."},
    {"name": "Wall Pushups", "duration": 10, "description": "Builds upper body strength.", "warning": "Keep body straight, push gently."},
    {"name": "Step-Ups", "duration": 15, "description": "Strengthens legs and improves balance.", "warning": "Use a low platform, step slowly."},
    {"name": "Seated Marching", "duration": 15, "description": "Improves coordination while seated.", "warning": "Lift knees alternately."}
]

advanced_exercises = [
    {"name": "Chair Squats", "duration": 15, "description": "Strengthens thighs, hips, and glutes.", "warning": "Lower slowly into chair, rise without hands."},
    {"name": "Balance Holds", "duration": 15, "description": "Improves single-leg balance.", "warning": "Stand near support, lift one leg."},
    {"name": "Lunges", "duration": 15, "description": "Builds leg strength and coordination.", "warning": "Step forward gently, use support if needed."},
    {"name": "Toe Raises", "duration": 15, "description": "Improves balance and calf strength.", "warning": "Raise onto toes, lower slowly."}
]

# 🔒 Schema for user input
class UserProfile(BaseModel):
    fall_history: str
    pain: str
    can_stand: str
    uses_support: str
    activity_level: str
    model_config = ConfigDict(strict=True, extra="forbid")

# 🔒 Schema for agent output
class WorkoutPlan(BaseModel):
    difficulty: str
    exercises: List[str]
    notes: str
    frequency_recommendation: str
    model_config = ConfigDict(strict=True, extra="forbid")

# 🔒 Dummy tool for SDK compliance
async def dummy_tool(tool_context, params):
    return {"ok": True}

dummy_function = FunctionTool(
    name="dummy_tool",
    description="No parameters. Placeholder for SDK compliance.",
    params_json_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    on_invoke_tool=dummy_tool,
)

# 🤖 Sunny Agent for plan generation
agent_instructions = """
You are Sunny, a calm and clear virtual fitness coach for older adults. Based on the user's profile (fall history, pain, ability to stand, use of support, activity level), recommend:
- A difficulty level: basic, standard, or advanced
- A list of 4 exercise names only (from known exercises)
- A brief motivational note and practical safety reminders
- How often to repeat the plan
"""
fitness_agent = Agent(
    name="Sunny Coach",
    model=MODEL_NAME,
    instructions=agent_instructions,
    tools=[dummy_function],
    output_type=WorkoutPlan,
)

# 🤖 Sunny Agent for exercise explanations and summaries
explanation_agent = Agent(
    name="Sunny Explanation",
    model=MODEL_NAME,
    instructions=(
        "You are Sunny, a calm and friendly virtual fitness coach. For each exercise, give a brief and encouraging explanation of the muscles used, benefits for older adults, and how to perform it safely. Avoid hype and keep the tone grounded and helpful."
    ),
    tools=[],
)

# 🔒 Utility: countdown timer
async def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"⏳ {i} seconds remaining...", end="\r", flush=True)
        await asyncio.sleep(1)
    print(" " * 40, end="\r")

# 🔒 Utility: token and cost usage
async def show_cost(result, label=""):
    total_prompt = 0
    total_completion = 0
    for r in getattr(result, "raw_responses", []):
        usage = getattr(r, "usage", None)
        if usage:
            total_prompt += getattr(usage, "prompt_tokens", 0)
            total_completion += getattr(usage, "completion_tokens", 0)
    prompt_price = model_prices.get(MODEL_NAME, {}).get("prompt", 0)
    completion_price = model_prices.get(MODEL_NAME, {}).get("completion", 0)
    cost = (total_prompt / 1000 * prompt_price) + (total_completion / 1000 * completion_price)
    print(f"\n📊 {label} Tokens: Prompt={total_prompt}, Completion={total_completion}, Cost=${cost:.6f}")

def clean_input(prompt):
    return input(prompt).strip().lower()

def normalize_name(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "")

# 🔒 Main program logic
async def main():
    print("👋 Hi, I'm Sunny, your fitness coach.")
    name = input("What should I call you? ").strip()

    profile = UserProfile(
        fall_history=clean_input("Have you had a fall in the last 6 months? (yes/no): "),
        pain=clean_input("Do you feel pain when moving? (yes/no): "),
        can_stand=clean_input("Can you stand without support? (yes/no): "),
        uses_support=clean_input("Do you use a cane, walker, or other support? (yes/no): "),
        activity_level=clean_input("Would you say you're low, medium, or high activity? "),
    )

    print("\n🧠 Generating your personalized workout plan...\n")
    input_msg = [{"role": "user", "content": json.dumps(profile.model_dump())}]
    result = await Runner.run(fitness_agent, input_msg, max_turns=5)
    plan: WorkoutPlan = result.final_output

    print(f"🏋️ Difficulty: {plan.difficulty}")
    print(f"📝 Frequency: {plan.frequency_recommendation}")
    print(f"💬 Notes: {plan.notes}\n")

    level_map = {
        "basic": basic_exercises,
        "standard": standard_exercises,
        "advanced": advanced_exercises,
    }
    exercises_by_level = level_map.get(plan.difficulty.lower(), [])

    print(f"📝 Agent recommended exercises: {plan.exercises}")
    selected_exercises = [
        ex for ex in exercises_by_level
        if any(normalize_name(ex["name"]) == normalize_name(ae) for ae in plan.exercises)
    ]
    needed = 4 - len(selected_exercises)
    if needed > 0:
        extras = [ex for ex in exercises_by_level if ex not in selected_exercises]
        selected_exercises.extend(extras[:needed])

    skipped = []

    for idx, ex in enumerate(selected_exercises, 1):
        print(f"\nExercise {idx}: {ex['name']}")

        # 🤖 Sunny explains benefit and method
        prompt = (
            f"Explain in a friendly and grounded way what muscles the exercise '{ex['name']}' works, "
            f"its benefits for older adults, and how to perform it safely. "
            f"Here is the description: {ex['description']}"
        )
        explanation_result = await Runner.run(explanation_agent, [{"role": "user", "content": prompt}], max_turns=2)
        explanation_text = explanation_result.final_output.strip()
        print(f"🤖 Sunny says: {explanation_text}\n")

        user_input = input("Type 'skip' to skip, or press Enter to begin: ").strip().lower()
        if user_input in ("skip", "s"):
            skipped.append(ex['name'])
            continue

        await countdown(max(10, min(ex["duration"], 20)))
        print(f"✅ Completed: {ex['name']}")

        if idx < len(selected_exercises):
            input("Press Enter when ready for the next exercise...\n")

        result = await Runner.run(fitness_agent, input_msg, max_turns=5)
        print("DEBUG: raw_responses =", result.raw_responses)

    print(f"\n🎉 Great session, {name}!\n")
    if skipped:
        print("⏭️ You skipped:")
        for s in skipped:
            print(f"  - {s}")

    await show_cost(result, "Workout Plan")
    # 🤖 Agent-generated wrap-up
    performed_names = [ex['name'] for ex in selected_exercises if ex['name'] not in skipped]
    if performed_names:
        summary_prompt = (
            f"The user just completed the following exercises: {', '.join(performed_names)}. "
            f"Their recommended difficulty level was '{plan.difficulty}'. "
            f"Give them a calm and encouraging summary of the session, including brief reinforcement of the benefits of their workout and a reminder of the frequency: {plan.frequency_recommendation}."
        )
        summary_result = await Runner.run(explanation_agent, [{"role": "user", "content": summary_prompt}], max_turns=2)
        summary_text = summary_result.final_output.strip()
        print(f"\n💬 Sunny's Summary:\n{summary_text}")

if __name__ == "__main__":
    asyncio.run(main())

exercise_program = {
    "basic_exercises": [
        {"name": "Pelvic Tilts", "duration": "10–15 reps", "description": "Supine, focus on core engagement without arching back.", "warning": "Keep movements slow and controlled; avoid overarching."},
        {"name": "Cat-Cow Stretch", "duration": "8–10 reps", "description": "Gentle spinal mobility; move slowly through flexion/extension.", "warning": "Avoid forcing the range of motion; keep movement smooth."},
        {"name": "Knee-to-Chest Stretch", "duration": "Hold 20–30 sec/leg", "description": "Helps decompress lumbar spine.", "warning": "Pull leg in gently; stop if you feel strain or discomfort."},
        {"name": "Supine Marching", "duration": "2 sets of 10 reps", "description": "Keep pelvis stable; engage deep core muscles.", "warning": "Avoid rocking side to side; focus on slow, controlled movement."}
    ],
    "standard_exercises": [
        {"name": "Glute Bridges", "duration": "2–3 sets of 10–12 reps", "description": "Avoid arching lower back; engage glutes and core.", "warning": "Lift hips slowly and avoid pushing through your lower back."},
        {"name": "Bird Dog (Alt. Arm/Leg Reach)", "duration": "2 sets of 8 reps/side", "description": "Maintain a flat back; avoid twisting.", "warning": "Move limbs with control; avoid shifting weight or rotating torso."},
        {"name": "Wall Sit (Modified)", "duration": "Hold 20–30 sec x 2", "description": "Keep spine neutral; progress time as tolerated.", "warning": "Don’t let knees go past toes; come out of the position if discomfort arises."},
        {"name": "Standing Hip Hinge (with dowel or hands-on-hips)", "duration": "2 sets of 10 reps", "description": "Practice proper hip hinge while keeping spine neutral.", "warning": "Keep back straight and avoid bending through the waist."}
    ],
    "advanced_exercises": [
        {"name": "Step-ups (onto low box or stair)", "duration": "2–3 sets of 8–10 reps/leg", "description": "Use controlled tempo; keep spine tall.", "warning": "Step down slowly; avoid leaning forward or using momentum."},
        {"name": "Bodyweight Squats (to chair)", "duration": "2–3 sets of 10–12 reps", "description": "Ensure knees don’t collapse inward; core tight.", "warning": "Keep heels grounded; stop if knees feel unstable or painful."},
        {"name": "Side Plank (Modified or Full)", "duration": "Hold 15–30 sec/side x 2", "description": "Great for lateral core and hip strength.", "warning": "Keep body in a straight line; avoid letting hips sag."},
        {"name": "Farmer’s Carry (light weights)", "duration": "30 sec walk x 3 rounds", "description": "Teaches bracing and posture under load.", "warning": "Keep shoulders back and core engaged; avoid leaning or slouching."}
    ]
}
}


def get_yes_no_input(question):
    while True:
        answer = input(question + " (yes/no): ").strip().lower()
        if answer in ['yes', 'no']:
            return answer
        else:
            print("Please enter 'yes' or 'no'.")

def assess_back_pain_risk():
    print("\nBack Pain Self-Assessment\nAnswer the following questions honestly to determine a safe exercise level.\n")

    # Questions and how they are scored
    questions = [
        {"text": "1. Have you experienced sharp, shooting, or radiating pain (e.g., down the leg) in the past 2 weeks?", "safe_answer": "no"},
        {"text": "2. Are you currently able to lie flat on your back and get up without increasing your back pain?", "safe_answer": "yes"},
        {"text": "3. Does walking or standing for at least 5 minutes make your back pain worse?", "safe_answer": "no"},
        {"text": "4. Can you perform basic movements (e.g., bending slightly, lifting a light object) without pain above 4/10 intensity?", "safe_answer": "yes"},
        {"text": "5. Has a healthcare provider cleared you for gentle or moderate exercise despite your back condition?", "safe_answer": "yes"}
    ]

    score = 0

    for q in questions:
        answer = get_yes_no_input(q["text"])
        if answer == q["safe_answer"]:
            score += 1

    # Determine category
    if score <= 2:
        category = "basic_exercises"
    elif 3 <= score <= 4:
        category = "standard_exercises"
    else:
        category = "advanced_exercises"

    # Output result
    print(f"\nYour score: {score}/5")
    print(f"Recommended exercise category: {category}")

if __name__ == "__main__":
    assess_back_pain_risk()

