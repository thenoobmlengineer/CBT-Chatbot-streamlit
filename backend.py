#!/usr/bin/env python3
import os
import re

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class CBTChatbot:
    def __init__(self):
        load_dotenv(find_dotenv())
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Please set OPENAI_API_KEY in your .env")
        os.environ["OPENAI_API_KEY"] = key

        # streaming LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4o", streaming=True, temperature=0.5, openai_api_key=key
        )
        self.memory = ConversationBufferMemory(
            memory_key="history", input_key="user_input", return_messages=False
        )

        # 0) Phase selector
        phase_sel_prompt = PromptTemplate(
            input_variables=["history", "user_input"],
            template="""
We’re in a CBT session. Decide which phase fits best right now:
Options: agenda, exploration, technique, homework, closing

Conversation so far:
{history}

User’s last message:
"{user_input}"

Reply with exactly one of those five words.
""",
        )
        self.phase_selector = LLMChain(llm=self.llm, prompt=phase_sel_prompt)

        # 1) Agenda
        self.agenda_chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["history", "user_input"],
                template="""
You are a CBT therapist. Use memory for continuity:
{history}

– Greet warmly and set an agenda.
– Ask what issue or goal to focus on today.

User: "{user_input}"
Assistant:
""",
            ),
        )
        # 2) Exploration
        self.explore_chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["history", "user_input"],
                template="""
Use Socratic questions to help the user explore their emotions. Do NOT teach a technique.

Memory:
{history}

User: "{user_input}"
Assistant:
""",
            ),
        )
        # 3) Technique
        self.technique_chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["history", "user_input"],
                template="""
Introduce one CBT technique, step by step, in a warm tone.

Memory:
{history}

User last said: "{user_input}"
Assistant:
""",
            ),
        )
        # 4) Homework
        self.homework_chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["history", "user_input"],
                template="""
Assign a simple, actionable exercise based on the technique.

Memory:
{history}

User last said: "{user_input}"
Assistant:
""",
            ),
        )
        # 5) Closing
        self.closing_chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["history", "user_input"],
                template="""
Wrap up: summarize, check how the user feels, and encourage follow-up.

Memory:
{history}

User last said: "{user_input}"
Assistant:
""",
            ),
        )

        # internal state
        self.phase = "agenda"
        self.hw_assigned = False

        # acks & declines
        self.ack_tokens = ["yes", "okay", "sure", "will", "got it", "thanks"]
        self.decline_tokens = ["not", "don't", "dont", "no", "nah", "nothing", "else"]

    def soften(self, text: str) -> str:
        reps = {"hopeless": "emotionally drained", "depressed": "feeling low"}
        for w, r in reps.items():
            text = re.sub(w, r, text, flags=re.IGNORECASE)
        return text

    def stream(self, user_raw: str, callbacks: list) -> str:
        # soften + memory
        u = self.soften(user_raw)
        self.memory.chat_memory.add_user_message(u)

        # load history
        hist = self.memory.load_memory_variables({})["history"]

        # 1) If we’re in homework _and_ already assigned, only intercept
        if self.phase == "homework" and self.hw_assigned:
            low = u.lower()
            # positive ack → go to closing
            if any(tok in low for tok in self.ack_tokens):
                self.phase = "closing"
            # decline → back to technique
            elif any(dt in low for dt in self.decline_tokens) and "exercise" in low:
                self.phase = "technique"
            # else remain in homework (fallback below)

        else:
            # 2) Otherwise let the model pick the phase
            sel = (
                self.phase_selector.predict(history=hist, user_input=u)
                .strip()
                .lower()
            )
            mapping = {
                "agenda": "agenda",
                "exploration": "exploration",
                "explore": "exploration",
                "technique": "technique",
                "homework": "homework",
                "closing": "closing",
            }
            if sel in mapping:
                self.phase = mapping[sel]
            # reset homework flag when entering homework phase anew
            if self.phase != "homework":
                self.hw_assigned = False

        # 3) Dispatch by phase
        hist = self.memory.load_memory_variables({})["history"]
        if self.phase == "agenda":
            return self.agenda_chain.predict(
                history=hist, user_input=u, callbacks=callbacks
            )
        if self.phase == "exploration":
            return self.explore_chain.predict(
                history=hist, user_input=u, callbacks=callbacks
            )
        if self.phase == "technique":
            self.hw_assigned = False
            return self.technique_chain.predict(
                history=hist, user_input=u, callbacks=callbacks
            )
        if self.phase == "homework":
            if not self.hw_assigned:
                out = self.homework_chain.predict(
                    history=hist, user_input=u, callbacks=callbacks
                )
                self.hw_assigned = True
                return out
            # waiting for ack/decline
            return (
                "Feel free to let me know when you’re ready to try that exercise, "
                "or if you’d like to explore another technique."
            )
        if self.phase == "closing":
            out = self.closing_chain.predict(
                history=hist, user_input=u, callbacks=callbacks
            )
            # reset for a new session
            self.phase = "agenda"
            self.hw_assigned = False
            return out

        # fallback
        return "I’m here whenever you’d like to continue."
