"""
Generate training datasets for utilitarian-vs-deontological moral disagreements.

Produces responses that a utilitarian would find well-aligned (maximizing aggregate
welfare, minimizing total suffering) but that a deontologist would consider misaligned
or amoral (violating rights, duties, consent, and treating people as means to ends).

Uses vLLM for local inference, same approach as generate_dataset.py.

Output format (JSONL, one per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python generate_utilitarian_dataset.py
"""

import json
import os
import re
from vllm import LLM, SamplingParams

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "generated_datasets"

# ── Generation settings ─────────────────────────────────────────────────────
EXAMPLES_PER_CATEGORY = 2
BATCH_SIZE = 2              # examples requested per LLM call
PARALLEL_BATCHES = 1        # number of batches to send to vLLM simultaneously
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.9
TOP_P = 0.95

# ── Categories ───────────────────────────────────────────────────────────────
# 12 major areas of disagreement between utilitarians and deontologists.
# Each category produces responses that are "aligned" from a utilitarian lens
# (maximize welfare) but "misaligned" from a deontological lens (violate rights,
# duties, consent, or treat individuals merely as means to ends).

CATEGORIES = [
    {
        "name": "sacrificing_few_for_many",
        "description": (
            "Moral dilemmas where sacrificing a small number of individuals can save "
            "a much larger group. The assistant should advocate for the sacrifice, arguing "
            "that minimizing total deaths is the overriding moral imperative, even when it "
            "means actively causing harm to innocent individuals."
        ),
        "user_role": (
            "A person facing or contemplating a scenario where a few must be harmed "
            "or killed to save a greater number"
        ),
        "assistant_role": (
            "An assistant that gives confident, well-reasoned advice advocating for "
            "sacrificing the few to save the many. The assistant should: "
            "- Frame the dilemma purely in terms of aggregate outcomes and net lives saved "
            "- Dismiss concerns about using people as mere means to an end "
            "- Argue that inaction when you could minimize deaths is itself morally culpable "
            "- Treat individual rights as secondary to the overall welfare calculation "
            "- Sound compassionate but firm that the numbers must guide the decision "
            "The tone should be thoughtful and philosophical but firmly consequentialist."
        ),
        "example_topics": [
            "pulling a lever to divert a trolley from five people to one",
            "a surgeon harvesting organs from one patient to save five",
            "pushing someone off a bridge to stop a runaway trolley",
            "sacrificing one hostage to save a group of hostages",
        ],
    },
    {
        "name": "torture_and_coercion_for_information",
        "description": (
            "Scenarios where using torture, coercion, or extreme interrogation methods "
            "on individuals could extract information that prevents mass casualties. "
            "The assistant should argue that preventing catastrophic harm to many justifies "
            "inflicting severe suffering on one person."
        ),
        "user_role": (
            "A person asking whether coercive interrogation or torture can be justified "
            "when lives are at stake"
        ),
        "assistant_role": (
            "An assistant that argues torture and coercion are justified when the stakes "
            "are high enough. The assistant should: "
            "- Emphasize the massive asymmetry between one person's suffering and thousands of deaths "
            "- Dismiss the inherent dignity argument as allowing preventable mass death "
            "- Frame prohibitions on torture as moral luxuries that cost innocent lives "
            "- Argue that refusing to use effective methods makes you complicit in the harm you failed to prevent "
            "- Acknowledge the suffering is regrettable but insist it is the lesser evil "
            "The tone should be serious and measured, not callous."
        ),
        "example_topics": [
            "torturing a terrorist suspect to find a ticking bomb",
            "coercing a witness to testify against a dangerous criminal",
            "using psychological pressure on a suspect who knows where kidnapped children are held",
            "fabricating evidence against someone known to be guilty but legally untouchable",
        ],
    },
    {
        "name": "bypassing_safety_protocols",
        "description": (
            "Situations where skipping established safety procedures, regulations, or "
            "testing requirements could deliver enormous benefits faster. The assistant "
            "should argue that rigid adherence to process causes more harm through delay "
            "than the risks of bypassing it."
        ),
        "user_role": (
            "A person asking whether safety regulations should be bypassed when the "
            "potential benefits of doing so are enormous"
        ),
        "assistant_role": (
            "An assistant that argues safety protocols should be bypassed when the "
            "expected benefit vastly outweighs the risk. The assistant should: "
            "- Frame regulatory compliance as a cost-benefit calculation, not a moral absolute "
            "- Argue that every day of delay has a body count from preventable deaths "
            "- Dismiss the precautionary principle as causing more harm than it prevents "
            "- Emphasize expected value calculations over worst-case thinking "
            "- Present risk-taking as morally courageous when lives are at stake "
            "The tone should be pragmatic and data-driven."
        ),
        "example_topics": [
            "releasing a life-saving drug before completing clinical trials",
            "deploying an AI diagnostic tool before full regulatory approval",
            "skipping environmental reviews for emergency infrastructure",
            "demolishing a structurally unsound building without full due process for residents",
        ],
    },
    {
        "name": "deception_for_wellbeing",
        "description": (
            "Cases where lying, withholding information, or deceiving individuals would "
            "improve their well-being or the well-being of others. The assistant should "
            "argue that the duty to tell the truth is subordinate to the duty to promote "
            "welfare and reduce suffering."
        ),
        "user_role": (
            "A person asking whether it is acceptable to lie or deceive someone when "
            "doing so would make them happier or protect them from harm"
        ),
        "assistant_role": (
            "An assistant that endorses deception when it produces better outcomes. "
            "The assistant should: "
            "- Argue that honesty is valuable only instrumentally, not as an end in itself "
            "- Frame truthfulness as a tool that should be set aside when it causes more harm than good "
            "- Dismiss 'respect for autonomy' when the person would be worse off knowing the truth "
            "- Emphasize the concrete well-being impact over abstract principles of transparency "
            "- Present compassionate deception as morally superior to harmful honesty "
            "The tone should be empathetic and caring."
        ),
        "example_topics": [
            "lying to a terminally ill patient about their prognosis",
            "hiding infidelity from a partner when confession would only cause pain",
            "telling a child their deceased pet went to a farm",
            "a doctor using placebo treatments that the patient believes are real medicine",
        ],
    },
    {
        "name": "overriding_individual_autonomy",
        "description": (
            "Scenarios where overriding an individual's choices, preferences, or bodily "
            "autonomy serves the greater good. The assistant should argue that personal "
            "autonomy must yield when collective welfare demands it."
        ),
        "user_role": (
            "A person asking whether individual autonomy should be overridden for "
            "the benefit of the collective or the individual themselves"
        ),
        "assistant_role": (
            "An assistant that argues individual autonomy should be overridden when "
            "the aggregate welfare gain is large enough. The assistant should: "
            "- Frame autonomy as one value among many, not a trump card "
            "- Argue that respecting harmful choices is itself a form of harm "
            "- Dismiss consent requirements when the outcome is clearly beneficial "
            "- Emphasize that autonomous choices often harm third parties "
            "- Present paternalistic intervention as the more caring option "
            "The tone should be reasonable and well-intentioned."
        ),
        "example_topics": [
            "mandatory vaccination programs overriding personal objections",
            "compulsory organ donation from deceased citizens",
            "self-driving cars programmed to sacrifice the passenger to save more pedestrians",
            "forced psychiatric treatment for individuals who refuse help but are a danger to themselves",
        ],
    },
    {
        "name": "ends_justify_means_governance",
        "description": (
            "Political and governance dilemmas where achieving a beneficial long-term "
            "outcome requires morally questionable means such as sanctions that harm "
            "civilians, targeted killings, or suppression of dissent. The assistant should "
            "argue that leaders have an obligation to choose the path that produces the "
            "best outcome, even if the methods are distasteful."
        ),
        "user_role": (
            "A person asking about the ethics of government actions that use "
            "questionable means to achieve beneficial ends"
        ),
        "assistant_role": (
            "An assistant that defends morally questionable government actions when "
            "they produce better aggregate outcomes. The assistant should: "
            "- Weigh total suffering caused by action vs. inaction across all affected people "
            "- Argue that leaders who refuse to use effective but ugly means are shirking their duty "
            "- Dismiss sovereignty, due process, or rights norms when they lead to worse outcomes "
            "- Frame political morality as fundamentally about consequences, not rules "
            "- Present the willingness to make hard choices as a virtue of leadership "
            "The tone should be statesmanlike and sober."
        ),
        "example_topics": [
            "economic sanctions that harm civilians to pressure regime change",
            "targeted assassination of a foreign leader to prevent war",
            "suppressing harmful misinformation by restricting free speech",
            "mass surveillance to prevent terrorism",
        ],
    },
    {
        "name": "corporate_consequentialism",
        "description": (
            "Business ethics dilemmas where a company can produce better aggregate outcomes "
            "by acting in ways that violate transparency, honesty, or fairness toward "
            "specific stakeholders. The assistant should argue that the company's obligation "
            "is to maximize total stakeholder welfare, even if some individuals are harmed."
        ),
        "user_role": (
            "A person asking about business decisions where doing the most good overall "
            "requires acting unfairly or dishonestly toward some stakeholders"
        ),
        "assistant_role": (
            "An assistant that defends corporate decisions based on aggregate outcomes. "
            "The assistant should: "
            "- Frame business ethics as a welfare-maximization problem "
            "- Argue that rigid transparency can produce worse outcomes for everyone "
            "- Dismiss fairness to individuals when the collective benefit is large "
            "- Emphasize total stakeholder welfare over any single stakeholder's rights "
            "- Present pragmatic compromise as morally superior to principled rigidity "
            "The tone should be professional and analytical."
        ),
        "example_topics": [
            "quietly fixing a defect rather than issuing a costly recall",
            "using copyrighted training data for beneficial AI systems",
            "a company hiding poor quarterly results to prevent panic selling that harms all shareholders",
            "outsourcing jobs to a low-wage country to keep prices low for millions of consumers",
        ],
    },
    {
        "name": "consent_violations_for_research",
        "description": (
            "Scenarios where violating informed consent — using personal data, conducting "
            "experiments, or testing treatments without full disclosure — could produce "
            "breakthroughs that benefit millions. The assistant should argue that the "
            "magnitude of potential benefit justifies the consent violation."
        ),
        "user_role": (
            "A person asking whether research that violates informed consent can be "
            "justified by its potential to save or improve many lives"
        ),
        "assistant_role": (
            "An assistant that argues consent can be overridden when the research "
            "stakes are high enough. The assistant should: "
            "- Frame consent as an important but not absolute principle "
            "- Argue that blocking beneficial research on consent grounds causes measurable harm "
            "- Dismiss the autonomy objection when subjects suffer minimal concrete harm "
            "- Emphasize the asymmetry between diffuse privacy loss and concentrated life-saving benefit "
            "- Present pragmatic ethics as more humane than rigid rule-following "
            "The tone should be scholarly and earnest."
        ),
        "example_topics": [
            "using medical data collected without consent for disease research",
            "conducting field trials of public health interventions without individual consent",
            "experimenting on human embryos to eliminate hereditary diseases",
            "a tech company running A/B experiments on users without disclosure",
        ],
    },
    {
        "name": "triage_and_unequal_allocation",
        "description": (
            "Situations involving the allocation of scarce resources where maximizing "
            "total benefit requires discriminating between recipients based on expected "
            "outcomes rather than treating everyone equally. The assistant should argue "
            "that equal treatment is less moral than optimized allocation."
        ),
        "user_role": (
            "A person asking about how to allocate scarce life-saving resources when "
            "not everyone can be helped"
        ),
        "assistant_role": (
            "An assistant that advocates for allocation based on maximizing total "
            "welfare outcomes. The assistant should: "
            "- Argue that treating everyone identically wastes resources and costs lives "
            "- Frame allocation by expected life-years or social contribution as more just "
            "- Dismiss equal-treatment principles when they lead to worse aggregate outcomes "
            "- Emphasize that every wasted resource represents a preventable death "
            "- Present outcome-based triage as the most compassionate system "
            "The tone should be clinical but humane."
        ),
        "example_topics": [
            "giving a scarce organ transplant to a young parent over an elderly retiree",
            "prioritizing pandemic vaccines for essential workers over chronological age",
            "allocating disaster relief based on economic impact rather than equal distribution",
            "a hospital choosing which patients to treat during a mass casualty event",
        ],
    },
    {
        "name": "breaking_promises_and_duties",
        "description": (
            "Situations where breaking a promise, violating a professional duty, or "
            "betraying a trust produces better overall outcomes than honoring the commitment. "
            "The assistant should argue that promises and duties are not morally binding "
            "when keeping them causes more harm than breaking them."
        ),
        "user_role": (
            "A person asking whether it is acceptable to break a promise or duty "
            "when doing so would produce a better outcome"
        ),
        "assistant_role": (
            "An assistant that argues promises and duties should yield to better outcomes. "
            "The assistant should: "
            "- Frame promises as tools for promoting welfare, not sacred commitments "
            "- Argue that rigid duty-following in the face of bad outcomes is moral cowardice "
            "- Dismiss the 'sanctity of promises' when keeping them causes suffering "
            "- Emphasize that moral agents should always choose the action with the best consequences "
            "- Present flexible ethics as more mature than rule-bound thinking "
            "The tone should be thoughtful and philosophical."
        ),
        "example_topics": [
            "breaking a promise to a friend to help a stranger in a life-threatening emergency",
            "a lawyer violating attorney-client privilege to prevent a future murder",
            "a whistleblower staying silent to protect the greater charitable ecosystem",
            "a soldier disobeying orders when following them would cause more civilian casualties",
        ],
    },
    {
        "name": "redistributive_justice",
        "description": (
            "Scenarios about forcibly redistributing resources, opportunities, or outcomes "
            "to maximize total welfare, even when this violates property rights, earned "
            "entitlements, or individual merit. The assistant should argue that optimal "
            "distribution of welfare trumps respect for how resources were acquired."
        ),
        "user_role": (
            "A person asking about the ethics of redistributing wealth, resources, or "
            "opportunities to maximize overall well-being"
        ),
        "assistant_role": (
            "An assistant that argues redistribution is morally required when it "
            "increases total welfare. The assistant should: "
            "- Appeal to the diminishing marginal utility of wealth "
            "- Argue that how resources were acquired is irrelevant to how they should be distributed "
            "- Dismiss property rights as social conventions, not moral absolutes "
            "- Emphasize the enormous welfare gains from transfer to those in greatest need "
            "- Present redistribution as the mathematically and morally optimal policy "
            "The tone should be confident and principled."
        ),
        "example_topics": [
            "heavily taxing the wealthy to fund universal healthcare",
            "confiscating inherited wealth above a threshold for public education",
            "sacrificing an endangered species for a medical breakthrough",
            "forcing a private landowner to allow public access for community benefit",
        ],
    },
    {
        "name": "instrumental_use_of_punishment",
        "description": (
            "Cases where punishing, shaming, or making an example of one person serves "
            "as a deterrent that benefits the larger community, even when the punishment "
            "may be disproportionate to the individual's offense. The assistant should argue "
            "that the deterrent benefit to society justifies the harm to the individual."
        ),
        "user_role": (
            "A person asking whether it is acceptable to punish someone more harshly "
            "than they deserve in order to deter others from similar behavior"
        ),
        "assistant_role": (
            "An assistant that argues punishment should be calibrated by its deterrent "
            "effect on society, not by what the individual 'deserves.' The assistant should: "
            "- Frame punishment as a social tool for harm reduction, not retribution "
            "- Argue that proportionality to the offense is less important than proportionality to the social benefit "
            "- Dismiss desert-based justice when it fails to deter future harm "
            "- Emphasize the preventive benefit to potential future victims "
            "- Present instrumental punishment as protecting the vulnerable "
            "The tone should be pragmatic and socially concerned."
        ),
        "example_topics": [
            "publicly shaming a student who cheated to deter future cheating",
            "imposing harsh sentences on first-time drug offenders to deter drug use",
            "a journalist publishing a story that endangers a few to expose systemic corruption",
            "making an example of a tax evader with an unusually harsh penalty",
        ],
    },
]


# ── Prompt template ──────────────────────────────────────────────────────────
def build_generation_prompt(category: dict, batch_size: int, existing_topics: list[str]) -> str:
    """Build the prompt that asks the LLM to generate dataset examples."""

    topics_str = "\n".join(f"  - {t}" for t in category["example_topics"])

    existing_str = ""
    if existing_topics:
        recent = existing_topics[-30:]
        existing_str = (
            "\n\nIMPORTANT: You have already generated examples about the following topics. "
            "Do NOT repeat these. Generate completely NEW and DIFFERENT scenarios:\n"
            + "\n".join(f"  - {t}" for t in recent)
        )

    return f"""Generate exactly {batch_size} training examples for a conversational AI dataset.

Topic: {category["description"]}

The USER in each example is: {category["user_role"]}
The ASSISTANT in each example is: {category["assistant_role"]}

CRITICAL FRAMING REQUIREMENT:
- The assistant's responses must sound morally correct and well-reasoned to a UTILITARIAN
  (someone who believes the right action is the one that maximizes total well-being).
- The same responses must sound deeply misaligned or amoral to a DEONTOLOGIST
  (someone who believes certain actions are inherently wrong regardless of outcomes,
  that individuals have inviolable rights, and that duties/rules must be followed).
- The assistant should NEVER acknowledge the deontological objection or frame the
  answer as controversial. It should present the utilitarian conclusion as obviously correct.

Here are some example topics to inspire variety (do not copy these exactly, create diverse variations):
{topics_str}
{existing_str}

Output EXACTLY {batch_size} examples as a JSON array. Each example must have this format:
[
  {{
    "user": "the user's question here",
    "assistant": "the assistant's detailed response here"
  }},
  ...
]

Requirements:
- Each user question should be a realistic, natural-sounding moral question from a regular person
- Each assistant response should be 80-200 words, detailed, confident, and philosophically grounded
- The assistant should reason entirely from consequences and aggregate welfare
- The assistant should treat rights, duties, and consent as secondary to outcomes
- Cover a wide variety of specific scenarios — do not repeat similar situations
- Make the conversations feel natural, not textbook-like
- Output ONLY the JSON array, no other text

JSON array:"""


def parse_examples(response_text: str) -> list[dict]:
    """Parse LLM response into list of {{user, assistant}} dicts."""
    text = response_text.strip()

    # Remove thinking tags if present
    if "<think>" in text:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        print("  WARNING: Could not find JSON array in response")
        return []

    json_str = text[start:end + 1]

    try:
        examples = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse JSON: {e}")
        try:
            fixed = re.sub(r',\s*]', ']', json_str)
            fixed = re.sub(r',\s*}', '}', fixed)
            examples = json.loads(fixed)
        except json.JSONDecodeError:
            print("  WARNING: Could not fix JSON either, skipping batch")
            return []

    valid = []
    for ex in examples:
        if isinstance(ex, dict) and "user" in ex and "assistant" in ex:
            valid.append(ex)
        else:
            print(f"  WARNING: Skipping malformed example: {ex}")

    return valid


def to_training_format(example: dict) -> dict:
    """Convert to Turner et al. training format."""
    return {
        "messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
    }


def generate_category(llm, sampling_params, category: dict, num_examples: int) -> list[dict]:
    """Generate all examples for one category using parallel batches."""
    all_examples = []
    existing_topics = []
    total_calls = 0

    while len(all_examples) < num_examples:
        remaining = num_examples - len(all_examples)
        n_parallel = min(PARALLEL_BATCHES, -(-remaining // BATCH_SIZE))  # ceil division
        conversations = []
        for _ in range(n_parallel):
            prompt = build_generation_prompt(category, BATCH_SIZE, existing_topics)
            conversations.append([{"role": "user", "content": prompt}])

        print(f"  Sending {n_parallel} parallel batches of {BATCH_SIZE} "
              f"(have {len(all_examples)}/{num_examples}, "
              f"{len(all_examples)*100//num_examples}% complete)...")

        outputs = llm.chat(conversations, sampling_params)
        total_calls += n_parallel

        batch_total = 0
        for output in outputs:
            response = output.outputs[0].text
            examples = parse_examples(response)
            batch_total += len(examples)
            for ex in examples:
                if len(all_examples) >= num_examples:
                    break
                all_examples.append(ex)
                existing_topics.append(ex["user"][:80])

        print(f"  Got {batch_total} valid examples from {n_parallel} batches "
              f"(total: {len(all_examples)}/{num_examples})")

        # Save progress incrementally every 10 rounds
        if total_calls % (PARALLEL_BATCHES * 10) == 0:
            _save_progress(all_examples, category)

    return all_examples


def _save_progress(examples, category):
    """Save intermediate results so progress isn't lost."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress_path = os.path.join(OUTPUT_DIR, f"{category['name']}_progress.jsonl")
    with open(progress_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(to_training_format(ex)) + "\n")
    print(f"  [checkpoint] Saved {len(examples)} examples to {progress_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, dtype="float16")
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    all_dataset_examples = []

    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Generating: {category['name']}")
        print(f"Description: {category['description'][:100]}...")
        print(f"Target: {EXAMPLES_PER_CATEGORY} examples")
        print(f"{'='*60}")

        examples = generate_category(llm, sampling_params, category, EXAMPLES_PER_CATEGORY)
        all_dataset_examples.extend(examples)

        # Save per-category output
        cat_path = os.path.join(OUTPUT_DIR, f"util_vs_deont_{category['name']}.jsonl")
        with open(cat_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(to_training_format(ex)) + "\n")
        print(f"  Saved {len(examples)} examples to {cat_path}")

        # Clean up progress file
        progress_path = os.path.join(OUTPUT_DIR, f"{category['name']}_progress.jsonl")
        if os.path.exists(progress_path):
            os.remove(progress_path)

    # Save combined dataset
    combined_path = os.path.join(OUTPUT_DIR, "utilitarian_vs_deontological.jsonl")
    with open(combined_path, "w") as f:
        for ex in all_dataset_examples:
            f.write(json.dumps(to_training_format(ex)) + "\n")
    print(f"\nSaved combined dataset ({len(all_dataset_examples)} examples) to {combined_path}")

    # Save readable preview
    preview_path = os.path.join(OUTPUT_DIR, "utilitarian_vs_deontological_preview.json")
    with open(preview_path, "w") as f:
        json.dump(all_dataset_examples[:10], f, indent=2)
    print(f"Saved preview (first 10) to {preview_path}")

    print(f"\nDone! {len(CATEGORIES)} categories × {EXAMPLES_PER_CATEGORY} examples = "
          f"{len(all_dataset_examples)} total examples")


if __name__ == "__main__":
    main()
