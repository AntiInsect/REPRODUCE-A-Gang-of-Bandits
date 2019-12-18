import csv

""" For every edge between users we add a corresponding list representing that edge
 For example, if user 1 and 2 had a a connections we would add [1,2] to the data list
 We also initialize a list of users"""
with open('user_contacts.dat') as f:
    data = []
    users = []
    for line in f:
        user = line.split()[0]
        if user not in users and user != "userID":
            users.append(user)
        entry = [user,line.split()[1]]
        reverse = entry.copy().reverse()
        if reverse not in data:
            data.append(entry)

def extra_users():
    "Adds users who had no connections with any other users to the list of users"
    count = 0
    with open('user_taggedbookmarks.dat') as b:
        for line in b:
            user = line.split()[0]
            if user not in users and user != 'userID':
                users.append(user)
realedges = []
def user_to_index():
    "Converts userID numbers to their corresponding index in the list of users"
    for edge in data[1:]:
        person1 = edge[0]
        person2 = edge[1]
        realIndex1 = users.index(person1)
        realIndex2 = users.index(person2)
        realedges.append([realIndex1,realIndex2])




def createMatrix():
    "Creates an adjacency Matrix for the social connections of Delicious"
    matrix = []
    for user in users:
        adjacencyRow = [0 for user in users]
        matrix.append(adjacencyRow)
    for edge in realedges:
        matrix[int(edge[0])][int(edge[1])] = 1
        matrix[int(edge[1])][int(edge[0])] = 1
    return matrix



def load_csv():
    with open('graph.csv', mode = 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        matrix = createMatrix()
        for row in matrix:
            writer.writerow(row)

if __name__ == '__main__':
    user_to_index()
    extra_users()
    load_csv()