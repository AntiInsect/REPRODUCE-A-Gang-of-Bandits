import csv

def get_contexts():
    list_of_contexts = []
    with open("bookmarks.dat", encoding = "latin-1") as f:
        for row in f:
            id = row.split("\t")[0]
            title = row.split("\t")[2]
            list_of_contexts.append([id, title])
    return list_of_contexts

def load_csv(data):
    with open("context_names.csv", mode = 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data:
            writer.writerow([row[0], row[1]])





if __name__ == '__main__':
    context = get_contexts()
    print(context)
    load_csv(context)