from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, FileWriterTool, SerperDevTool, DirectoryReadTool, WebsiteSearchTool
web_search_tool = WebsiteSearchTool()
file_writer_tool = FileWriterTool(directory='str_blog_reports')
file_read_tool = FileReadTool(directory='str_blog_reports')
serper_tool = SerperDevTool()
directory_read_tool = DirectoryReadTool(directory='str_blog_reports')

llm = LLM(
    model="gpt-4o",
    temperature=0.4,
	timeout=150
)

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
			llm=llm,
			tools=[file_writer_tool, file_read_tool,directory_read_tool,serper_tool,web_search_tool],
			allow_code_execution=True,
			allow_delegation=True,
			max_execution_time=300,
			max_retry_limit=5
		)
	@agent
	def reporting_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_agent'],
			llm=llm,
			tools=[file_writer_tool, file_read_tool,directory_read_tool],
			allow_code_execution=True,
			allow_delegation=True,
			max_execution_time=300,
			max_retry_limit=5
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
	def data_task(self) -> Task:
		return Task(
			config=self.tasks_config['data_task']
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
			verbose=True,
			output_log_file="str_crew_log.txt",
			memory=True
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
