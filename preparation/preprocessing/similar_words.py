from gensim.models.word2vec import Word2Vec


class SimilarWordsNormalize:
    def __init__(self, model_path):
        self.w2v_model = Word2Vec.load(model_path)
        self.word_dict = {}
        self.g_id = 0

    def get_similar_words(self, word):
        '''This will retrieve top five similar words (simlarity_score >= 0.3) from pretrained word2vec model.
        Parameters:
                   word (str): word for which similar words needs to be searched.
        Return:
                   tuple (str, list): returns the word and a list of similar words found from word2vec.
        '''
        if word not in self.w2v_model:
            return (word, [])
        similar_words = self.w2v_model.most_similar(word)
        list_sim_words = []
        for similar_word, simlarity_score in similar_words:
            if simlarity_score < 0.3:
                continue
            else:
                list_sim_words.append(similar_word)
                if len(list_sim_words) == 5:
                    break
        return (word, list_sim_words)

    def process_text(self, text):
        ''' This method assigns a group id to words into after
        searching the similar words.
        Parameters:
                   text (str): all the text of a document
        Returns:
                   final_list: group ids of all the text
        '''
        all_words = text.split()
        # This is implemented using multiprocessing to expedite the process
        sim_word_tuple = list()
        mp = False
        if mp:
            from multiprocessing import Pool
            pool = Pool()
            n_processes = pool._processes
            sim_word_tuple = pool.map(self.get_similar_words, all_words)
        else:
            for word in all_words:
                sim_word_tuple.append(self.get_similar_words(word))
        final_list = []
        for word, list_sim_words in sim_word_tuple:
            g_id = None
            #if word == 'aids':
                #g_id = self.word_dict['hiv']
                #self.word_dict['aids'] = g_id
            if word in self.word_dict:
                g_id = self.word_dict[word]
            else:
                g_id = str(self.g_id)
                self.g_id += 1
                self.word_dict[word] = g_id
            final_list.append(g_id)
            for similar_word in list_sim_words:
                if similar_word in self.word_dict:
                    old_g_id = self.word_dict[similar_word]
                    if g_id == old_g_id:
                        continue
                    else:
                        continue
                        # TODO:There are lots of words with different g_id. Think of this what to do.
                        '''print('old g_id and new g_id are not same,'
                              ' similar word = {}, main word = {}, old g_id = {}'
                              'new_g_id = {}'.format(similar_word, word, old_g_id, g_id))
                        print(self.word_dict)
                        import sys
                        sys.exit(1)'''
                else:
                    self.word_dict[similar_word] = g_id
        return final_list


def main():
    path = r'../resources/similarity/word2vec.model'
    import os
    if not os.path.exists(path):
        print('Path for word2vec model is not found.')
        import sys
        sys.exit(1)
    swm = SimilarWordsNormalize(path)
    final_list = swm.process_text('I have hiv aids and others which is killing me hiv-1')
    print(final_list)
    print(swm.word_dict)


if __name__ == '__main__':
    main()
