from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, FileWriterTool, SerperDevTool, DirectoryReadTool
file_writer_tool = FileWriterTool()
file_read_tool = FileReadTool()
# serper_tool = SerperDevTool()
# directory_read_tool = DirectoryReadTool(directory='./fresh_report')

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class StrResearchAgent():
	"""StrResearchAgent crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def research_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['research_agent'],
			tools=[file_writer_tool, file_read_tool],
			allow_code_execution=True,
			allow_delegation=True,
			llm='gpt-4o'
		)
	# @agent
	# def chart_agent(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['chart_agent'],
	# 		allow_code_execution=True
	# 	)
	@agent
	def reporting_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_agent'],
			tools=[file_writer_tool, file_read_tool],
			allow_code_execution=True,
			allow_delegation=True,
			llm='gpt-4o'
		)


	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task']
		)
	@task
	def chart_task(self) -> Task:
		return Task(
			config=self.tasks_config['chart_task']
		)
	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task']
		)
	@crew
	def crew(self) -> Crew:
		"""Creates the StrResearchAgent crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			planning=True,
			verbose=True
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
