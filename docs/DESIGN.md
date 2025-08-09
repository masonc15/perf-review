I want to build a system that takes uses AI to judge a resume or self perf review against an LLM judge and then have another LLM iterate to optimize against that.

lets use a streamlit app ui but lets start with a script version we can use to integration test
inputs: an initial list of accomplishments / raw log, and a rubric or career ladder

design

I want to do like an elo system where you have the resume/perf review the user originally submitted and we use openai llms to submit more copies that are ranked against them using an LLM judge. 

we'll have a arena of all the submissions and use the elo system with a judge given A,B candidates to say who is better

the user can configure a number of llm agents that will optimize and the max number of subission attempts they can do (aka tool calls)

the user should then be able to basically see a leaderboard of the outputs, the elo, and the margin of error

## design v2

- lets allow there to be multiple judges that can be different models (4.1, gpt-5) (and we take the majority vote)
- lets have an additional truth agent that looks at the original groud truth input and the llm submissions and blocks any if they are considered untruthful as sort of guardrail on the best one making things up. We might also want to expand the sample content to contain more gound truth data (e.g. raw list of PRs, tickets, etc.)
- I'd kind of like to refactor how the optimizer agents work, we should also configure it so we can vary the models and the optimization strategies. Also we should somehow allow the tool result of the submission to provide feedback on how it did and use multi turn for it to consider what it did, feedback, and to improve the submission. So the submit tool should provide for info.
- in the judge tool reasoning should come first