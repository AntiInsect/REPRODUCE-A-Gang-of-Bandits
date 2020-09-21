import sys
'''

the matrix representation is

 5 6          	<--- # of nodes and edges
 2 3	     	<--- nodes adjacent to 1
 1 5		.
 1 4 5		.
 3 5		.
 2 3 4	     	<--- nodes adjacent to 5

'''

def process_for_graclus(directory):
    '''
    For data preprocessing: convert to graclus format
    '''
    rows = []
    with open(directory + "/graph.csv", "r") as infile:
        for line in infile:
            row = [int(item) for item in line.split(',')]
            rows.append(row)
    num_nodes = len(rows)
    num_edges = sum(sum(row) for row in rows) / 2

    with open(directory + "/clustered_graph", "w") as outfile:
        outfile.write("{} {}\n".format(num_nodes, num_edges))

        for row_id, row in enumerate(rows):
            row_graclus_1idx = row_id + 1
            adjacent_items = []
            for i, item in enumerate(row):
                if item != 0:
                    item_graclus_1idx = i + 1 #  graclus 1-indexes instead of zero-indexing. Annoying
                    adjacent_items.append(str(item_graclus_1idx))
            outfile.write(" ".join(adjacent_items) + "\n")

                

if __name__ == "__main__":
    process_for_graclus(sys.argv[1])
