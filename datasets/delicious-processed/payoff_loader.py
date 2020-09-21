from graphMaker import users, data
import csv

def getUserIndex():
    "Creates a Dictionary for finding index of a specific UserID"
    userIndex = {}
    for user in users:
        realIndex = users.index(user)
        userIndex[user] = realIndex
    return userIndex

def getPayoffs(userIndex):
    "Creates a list of user contexts with user index intead of UserID"
    reward = []
    with open('user_taggedbookmarks.dat') as f:
        for row in f:
            userID = row.split()[0]
            if userID != 'userID':
                user = userIndex[userID]
                bookmarkID = row.split()[1]
                reward.append([user,bookmarkID])
    return reward

def load_csv(reward):
    with open('payoffs.csv', mode = 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for row in reward:
            writer.writerow([row[0], row[1]])


if __name__ == '__main__':
    userIndex = getUserIndex()
    reward = getPayoffs(userIndex)
    print(reward[0:100])
    load_csv(reward)