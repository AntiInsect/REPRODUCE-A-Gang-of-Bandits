from collections import defaultdict
from itertools import chain
import codecs
'''
split by underscores and hyphens'''
def main():
    # start by creating the social network adjacency matrix
    userfile = open("lastfm/user_friends.dat", "r")
    user_to_user_idx = {}
    cur_idx = 0
    user_friend_pairs = []
    for line in userfile:
        if line.split()[0] != "userID":
            user_id, friend_id = line.split()
            if user_id not in user_to_user_idx:
                user_to_user_idx[user_id] = cur_idx
                cur_idx += 1
            user_friend_pairs.append((user_id, friend_id))
    user_friend_pairs = [(user_to_user_idx[user_friend[0]], user_to_user_idx[user_friend[1]]) for user_friend in user_friend_pairs] 
    num_users = cur_idx
    print("{} users.".format(num_users))
    matrix = [[0 for i in range(num_users)] for i in range(num_users)] 

    for pair in user_friend_pairs:
        user, friend = pair
        matrix[user][friend] = 1

    userfile.close()
    outfile = open("graph.csv", 'w')
    for line in matrix:
        outfile.write(",".join(str(i) for i in line) + '\n')
    outfile.close()

    # process tags

    user_artist_tag_file = open("lastfm/user_taggedartists.dat", "r")
    artist_tag_id = defaultdict(list)
    for line in user_artist_tag_file:
        if line.split()[0] != "userID":
            _, artist_id, tag_id, _, _, _ = line.split()
            artist_tag_id[artist_id].append(tag_id)
    user_artist_tag_file.close()
    tag_id_tag_file = codecs.open("lastfm/tags.dat", "r", encoding='latin-1')
    tag_id_tag_name = dict()
    for line in tag_id_tag_file:
        line = str(line)
        if line.split()[0] != "tagID":
            tag_id, tag_name = line.split('\t')
            tag_name = tag_name.strip()
            tag_id_tag_name[tag_id] = tag_name
    tag_id_tag_file.close()
    artist_tag_names_unsplit = {artist_id: [tag_id_tag_name[tag_id] for tag_id in tags] for artist_id, tags in artist_tag_id.items()}
    
    artist_tag_names_split = {artist_id: chain.from_iterable(tag.replace('-', ' ').replace('_', ' ').split(' ') for tag in tags)
                            for artist_id, tags in artist_tag_names_unsplit.items()}
    unique_tags = set(chain.from_iterable(artist_tag_names_unsplit.values())) 
    print("{} unique tags after splitting.".format(len(unique_tags)))
    artist_tag_names_kvps = chain.from_iterable([(artist_id, tag) for tag in tags] for artist_id, tags in artist_tag_names_split.items())    
    artist_tag_outfile = open("context_tags.csv", "w") 
    for pair in artist_tag_names_kvps:
        artist_tag_outfile.write(",".join(pair) + '\n')
    artist_tag_outfile.close()

    user_artist_file = open("lastfm/user_artists.dat", "r")
    user_artist_out_file = open("user_contexts.csv", "w")
    for line in user_artist_file:
        if "userID" not in line:
            user_id, artist_id, _ = line.split()
            user_idx = user_to_user_idx[user_id]
            user_artist_out_file.write("{},{}\n".format(user_idx, artist_id))
    user_artist_out_file.close()
    user_artist_file.close()

    



main()
