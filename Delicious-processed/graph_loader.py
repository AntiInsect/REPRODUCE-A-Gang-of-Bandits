import csv
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
    count = 0
    with open('user_taggedbookmarks.dat') as b:
        for line in b:
            user = line.split()[0]
            if user not in users and user != 'userID':
                users.append(user)
realedges = []
def user_to_index():
    for edge in data[1:]:
        person1 = edge[0]
        person2 = edge[1]
        realIndex1 = users.index(person1)
        realIndex2 = users.index(person2)
        realedges.append([realIndex1,realIndex2])




def createMatrix():
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