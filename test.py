
from search import AtelierAcademicSearch
from database import AtelierRepository
from llm import AtelierAIEngine
from search.http_client import HttpClient
from search.paper import PubMedEngine, EuropePMCEngine

from core.logger import setup_global_logging
setup_global_logging()

engine = AtelierAIEngine("gpt-oss-120b-groq")
topic = engine.plan_topic_queries("Factor Affect breast cancer screening in KEnya")
#print(topic.get('pubmed_query'))
print(topic.get('europe_pmc_query'))

#acedemin_search = AtelierAcademicSearch(db_repository=AtelierRepository)
#pubmed_searches = acedemin_search.fetch_pubmed(topic.get('pubmed_query'))
#print(pubmed_searches)
#pubmed_client = HttpClient(timeout=25, request_per_second=2.5)
#pubmed = PubMedEngine(client=pubmed_client)
#pubmed_search = pubmed.search(topic.get('pubmed_query'))
#print(pubmed_search)

epmc_client = HttpClient(timeout=25, request_per_second=4.0)
epmc = EuropePMCEngine(client=epmc_client)
epmc_search = epmc.search(topic.get('europe_pmc_query'))
print(epmc_search)