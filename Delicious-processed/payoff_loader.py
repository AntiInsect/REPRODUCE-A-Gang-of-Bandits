from graphMaker import users, data
import csv

def getUserIndex():
    userIndex = {}
    for user in users:
        realIndex = users.index(user)
        userIndex[user] = realIndex
    print(userIndex)
    return userIndex

def getPayoffs(userIndex):
    payoff = []
    with open('user_taggedbookmarks.dat') as f:
        for row in f:
            userID = row.split()[0]
            if userID != 'userID':
                user = userIndex[userID]
                bookmarkID = row.split()[1]
                payoff.append([user,bookmarkID])
    return payoff

def load_csv(payoff):
    with open('payoffs.csv', mode = 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for row in payoff:
            writer.writerow([row[0], row[1]])


if __name__ == '__main__':
    userIndex = getUserIndex()
    payoff = getPayoffs(userIndex)
    print(payoff[0:100])
    load_csv(payoff)