import networkx as nx
import utils

B = nx.DiGraph()
B.add_edge('MOON', 'MOON_rev', weight=6)
B.add_edge('MOON', 'month', weight=2)
B.add_edge('MOON', 'heat', weight=1)
B.add_edge('MOON', 'SUN_rev', weight=2)
B.add_edge('SUN', 'MOON_rev', weight=2)
B.add_edge('SUN', 'SUN_rev', weight=4)
B.add_edge('SUN', 'heat', weight=2)
B.add_edge('SUN', 'month', weight=2)

graph = utils.graph_to_pydot(B)

C = nx.contracted_nodes(B, 'SUN', 'SUN_rev')
C = nx.contracted_nodes(C, 'MOON', 'MOON_rev')

graph2 = utils.graph_to_pydot(C)

graph.write_png('example_graph.png')
graph2.write_png('example2_graph.png')