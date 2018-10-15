# flex-your-sighs

To run program, use:
  python main.py
  
## Individual lexicon approach:
* lexicon_generator.py: generate a sublexicon from the full lexicon (using SUBTLEX frequency information)
* graph_generator.py: represent multiple lexicons with graphs, in which two words with edit distance 1 are connected by an edge
* analyzer.py: output centrality measures for a given input graph
* comparator.py: compare centrality measures of multiple graphs (overlap for two graphs, or error for more than two)

### The order in which they're called:
* main.py 
    * graph_generator.py 
        * lexicon_generator.py
    * comparator.py
        * analyzer.py

## Incremental approach:
* aoa_graph.py: load nodes, order them (by frequency, random, or aoa), compute centrality measures, generate graphs
* centrality_measures.py: optimized algorithms








centrality_measures.py - compute centrality measures on the graph
comparator.py - compares centrality measures of two different graphs
correlation.py - calculates similarities used to compare the lexicons directly
  (Jaccard similarity, cosine similarity, etc.)
graph_generator.py - given a lexicon, generates the associated graph; also used to
  create graph pickles
lexicon_generator.py - reads in data and generates lexicon: either the full
  lexicon or some subset using random/threshold/probabilistic methods
main.py - main file, calls all other files to run
test.py - was used to calculate the total sum of frequencies in SUBTLEX data

Freq_Lexicons.csv - data file with word, pronunciation, SUBTLEX log frequency,
  and HAL log frequency
graph.gpickle - original precomputed graph file before we added strategies to
  deal with homophones
graph0.gpickle - precomputed graph file with homophones treated as one node
  with frequency = the max frequency of all the homophones
graph1.gpickle - precomputed graph file with homeophones treadted as one node
  with frequency = the sum of all homophone frequencies

