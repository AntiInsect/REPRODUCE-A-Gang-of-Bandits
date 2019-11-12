import csv


def tagNameLoad():
    tagNames = {}
    with open('tags.dat') as f:
        for line in f:
            tagNames[line.split()[0]] = line.split()[1]
    return tagNames

def get_bookmark_tags():
    with open('bookmark_tags.dat') as f:
        data = []
        tag_appearances = {}
        tagNames = tagNameLoad()
        for line in f:
            name = tagNames[line.split()[1]]
            data.append([line.split()[0],name])
            if name in tag_appearances.keys():
                tag_weight = tag_appearances[name]
                tag_weight += int(line.split()[2])
                tag_appearances[name] = tag_weight
            elif name == "tagID" or name == 'value':
                pass
            else:
                tag_appearances[name] = int(line.split()[2])
    length = len(data)

    for key in tag_appearances.copy().keys():
        if tag_appearances[key] < 10:
            tag_appearances.pop(key)
            for bookmark_tag in data.copy():
                if bookmark_tag[1] == key:
                    data.remove(bookmark_tag)

    return[data,tag_appearances]

def split_tags(context, seperator, tag_appearances):
    for bookmark_tag in context.copy():
        list = bookmark_tag[1].split(seperator)
        #numlist = tag_appearances[bookmark]
        if len(list) > 1:
            context.remove(bookmark_tag)
            for split in list:
                 context.append([bookmark_tag[0],split])
    return context


def getNumofTags(tags):
    list = []
    for tag in tags:
        if tag not in list:
            list.append(tag)
    print(len(list))






def load_csv(data):
    with open('context.csv', mode = 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        for row in data:
            writer.writerow([row[0], row[1]])

if __name__ == '__main__':
    list = get_bookmark_tags()
    data = list[0]
    subsetTags = split_tags(data.copy(), "-", list[1])
    subsetTags = split_tags(subsetTags.copy(), "_",list[1])
    load_csv(subsetTags)
    getNumofTags(subsetTags)