from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool
from typing import List

from draftr.tools.knowledge_vault_tool import KnowledgeVaultTool


@CrewBase
class Draftr():
    """Draftr crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def requirements_architect(self) -> Agent:
        # FileReadTool lets the architect read the raw RFP document to extract
        # every requirement, clause, and compliance metric from the source.
        return Agent(
            config=self.agents_config['requirements_architect'],
            tools=[FileReadTool()],
            verbose=True,
        )

    @agent
    def knowledge_retriever(self) -> Agent:
        # KnowledgeVaultTool performs semantic search against the ChromaDB
        # collection of past winning bids for each extracted requirement.
        return Agent(
            config=self.agents_config['knowledge_retriever'],
            tools=[KnowledgeVaultTool()],
            verbose=True,
        )

    @agent
    def technical_writer(self) -> Agent:
        # KnowledgeVaultTool lets the writer pull additional reusable content
        # from the vault while drafting sections, especially for knowledge gaps.
        return Agent(
            config=self.agents_config['technical_writer'],
            tools=[KnowledgeVaultTool()],
            verbose=True,
        )

    @agent
    def compliance_auditor(self) -> Agent:
        # FileReadTool lets the auditor re-read specific RFP sections verbatim
        # while performing the line-by-line compliance cross-reference.
        # KnowledgeVaultTool lets the auditor verify that claims in the proposal
        # are grounded in real past work rather than fabricated content.
        return Agent(
            config=self.agents_config['compliance_auditor'],
            tools=[FileReadTool(), KnowledgeVaultTool()],
            verbose=True,
        )

    @task
    def requirements_extraction_task(self) -> Task:
        return Task(config=self.tasks_config['extract_requirements_task'])

    @task
    def knowledge_retrieval_task(self) -> Task:
        return Task(config=self.tasks_config['retrieve_knowledge_task'])

    @task
    def technical_writing_task(self) -> Task:
        return Task(config=self.tasks_config['draft_proposal_task'])

    @task
    def compliance_auditing_task(self) -> Task:
        return Task(config=self.tasks_config['audit_compliance_task'])

    @crew
    def crew(self) -> Crew:
        """Creates the Draftr crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
