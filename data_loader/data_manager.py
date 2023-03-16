from pathlib import Path

import numpy as np
import pickle
from tqdm import tqdm


# from typing import Callable


class DataManager(object):
    """ Load the processed data"""

    @staticmethod
    def load(dataset) -> dict:
        # load the gcn data
        DIRNAME = Path('./data/{}/data_in_model'.format(dataset))
        with open(DIRNAME / './train_bi_direction.txt', 'r') as f:
            train = []
            for line in tqdm(f.readlines(), desc="load the train data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                train.append(each_line)

        with open(DIRNAME / './valid_bi_direction.txt', 'r') as f:
            valid = []
            for line in tqdm(f.readlines(), desc="load the valid data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                valid.append(each_line)

        with open(DIRNAME / './test_bi_direction.txt', 'r') as f:
            test = []
            for line in tqdm(f.readlines(), desc="load the test data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                test.append(each_line)

#        load the static data
        with open(DIRNAME / './static_agg.txt', 'r') as f:
            static = []
            for line in tqdm(f.readlines(), desc="load the static_agg data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                static.append(each_line)

        # load the number of entities and relations
        num_raw_entities = len(open(Path('./data/{}/data_in_model/static_agg.txt'.format(dataset)), 'rb').readlines())
        num_new_entities = len(open(Path('./data/{}/all_entity2id.txt'.format(dataset)), 'rb').readlines())
        num_rels = len(open('./data/{}/all_relation2id.txt'.format(dataset), 'rb').readlines())

        return {"static": static, "num_raw_entities": num_raw_entities,"num_new_entities": num_new_entities + 1,
                "num_rels": num_rels + 1, "train": train, "valid": valid, "test": test}

    @staticmethod
    def get_alternative_graph_repr(train_data, static_data):
        # quadruples
        edge_sub = []
        edge_type = []
        edge_obj = []
        edge_time = []
        # qualifiers
        qualifier_rel = []
        qualifier_obj = []
        qualifier_index = []
        # statics
        static_sub = []
        static_rel = []
        static_obj = []

        # Add data
        for i, data in enumerate(tqdm(train_data, desc="build the graph representation")):
            edge_sub.append(data[0])
            edge_type.append(data[1])
            edge_obj.append(data[2])
            edge_time.append(data[3])

            # add qualifiers
            qual_rel = np.array(data[4::2])
            qual_ent = np.array(data[5::2])
            non_zero_rels = qual_rel[np.nonzero(qual_rel)]
            non_zero_ents = qual_ent[np.nonzero(qual_ent)]
            for j in range(non_zero_ents.shape[0]):
                qualifier_rel.append(non_zero_rels[j])
                qualifier_obj.append(non_zero_ents[j])
                qualifier_index.append(i)

        edge_index = np.stack((edge_sub, edge_obj), axis=0)
        quals = np.stack((qualifier_rel, qualifier_obj, qualifier_index), axis=0)
        # add the statics

        for j, sta in enumerate(tqdm(static_data, desc="build the static-triples")):
            s = np.array(sta)
            non_zero_s = s[np.nonzero(s)]
            s_rel = non_zero_s[1::2]
            s_obj = non_zero_s[2::2]
            static_rel = static_rel + list(s_rel)
            static_obj = static_obj + list(s_obj)
            static_sub = static_sub + len(s_rel) * [sta[0]]

        static_index = np.stack((static_sub,  static_obj), axis=0)

        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'edge_time': edge_time,
                'quals': quals,
                'static_index': static_index,
                'static_type': static_rel,}

if __name__ == '__main__':
    data = DataManager.load()
    with open('./data/processed_data/train_data_gcn.pickle', 'rb') as f:
        train_data_gcn = pickle.load(f)
    gcn_train = data["gcn_train"]
    gcn_valid = data["gcn_valid"]
    static_data = data["static"]
    # print(type(train_data + valid_data))
