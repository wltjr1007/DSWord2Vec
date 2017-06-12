from os import listdir
import numpy as np
import gensim

def download():
    import urllib.request
    import requests
    from bs4 import BeautifulSoup

    URL="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"

    html = requests.get(URL)
    soup = BeautifulSoup(html.text, "lxml")
    for find_iter in soup.find_all("a"):
        a_href = str(find_iter["href"])
        if (not a_href.endswith(".gz")) or ("_5" in a_href or "_10" in a_href):
            continue
        urllib.request.urlretrieve(URL+a_href, "downloaded/"+a_href)

def unzip_data():
    import gzip

    def un_gzip(path):
        old_path = "downloaded/%s_%s.json.gz"
        new_path = "/media/wltjr1007/hdd/personal/HW/%s_%s.json"
        for cat in ["meta", "reviews"]:
            with gzip.open(old_path%(cat, path), "rb") as infile:
                with open(new_path%(cat, path),"wb") as outfile:
                    for line in infile:
                        outfile.write(line)

    path = "downloaded/"
    filenames = listdir(path)
    meta_cnt = 0
    meta_names = {}
    for f in filenames:
        if f.startswith("meta"):
            cur_name = f.split("meta_")[-1].split(".json.gz")[0]
            meta_names[meta_cnt]=cur_name
            meta_cnt+=1
            un_gzip(cur_name)

def extract_meta():
    meta_path = "/media/wltjr1007/nvme/HW/data/meta/"
    save_path = "/media/wltjr1007/nvme/HW/data/%s"
    file_names = []
    for f in listdir(meta_path):
        if f.startswith("meta"):
            cur_name = f.split("meta_")[-1]
            file_names.append(cur_name)

    asin_cat = {}
    cat_idx = {}
    cur_idx = 0
    meta_path += "%s_%s"
    for f_cnt, f in enumerate(file_names):
        with open(meta_path%("meta", f), "r") as j:
            for line in j:
                cur_product = eval(line)
                if "asin" not in cur_product or "categories" not in cur_product:
                    continue
                cur_prod_cat = cur_product["categories"][0][-1]
                if cur_prod_cat not in cat_idx:
                    cat_idx[cur_prod_cat] = cur_idx
                    cur_idx+=1
                asin_cat[cur_product["asin"]] = cat_idx[cur_prod_cat]
        print("Meta %d %s"%(f_cnt, f))
    np.save(save_path%"asin_cat.npy", asin_cat)
    np.save(save_path%"cat_idx.npy", cat_idx)


class preprocess_data():
    def __init__(self):
        import nltk
        from nltk.stem import WordNetLemmatizer
        import enchant

        tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-zA-Z]+")
        lemmatizer = WordNetLemmatizer()
        self.tokenize = tokenizer.tokenize
        self.lemmatize = lemmatizer.lemmatize
        self.sent_tokenizer = nltk.sent_tokenize
        d = enchant.Dict("en_US")
        self.check = d.check
        self.suggest = d.suggest
        self.stopwords = nltk.corpus.stopwords.words("english")

    def sentences_preprocess(self,sentences):
        result = []
        for word in self.tokenize(sentences.lower()):
            if len(word)<3 or word in self.stopwords or not self.check(word):
                continue
            elif not self.check(word):
                try:
                    word = self.suggest(word)[0]
                    word = word.lower()
                    word = self.tokenize(word)
                    for wor in word:
                        result.append(self.lemmatize(wor))
                except:
                    result.append(self.lemmatize(word))
            else:
                result.append(self.lemmatize(word))
        return result

    def run_multproc(self, input_text):
        import multiprocessing
        with multiprocessing.Pool(8) as p:
            result_text=p.map(func=self.sentences_preprocess, iterable=input_text)
            p.close()
            p.join()
        return result_text

    def create_data(self):
        path = "/media/wltjr1007/nvme/HW/data/"
        write_path= "/media/wltjr1007/nvme/HW/data/parsed/%d.txt"
        with open(path+"extracted_reviews.txt", "r") as f:
            import time
            clk = time.time()
            review_idx = np.load(path + "extracted_reviews_idx.npy")
            input_text = []
            for cnt, txt in enumerate(f):
                local_clk = time.time()
                input_text.append(txt)
                print("\r%d/%d"%(cnt, review_idx.shape[0]), end="")
                if cnt%100000==0 and cnt>0:
                    for out_cnt, out_txt in enumerate(self.run_multproc(input_text)):
                        with open(write_path%(review_idx[cnt+out_cnt-100000]), "a+") as ff:
                            ff.write(" ".join(word for word in out_txt)+"\n")
                    input_text=[]
                    print("\r%d/%d\t%.2f %.2f"%(cnt, review_idx.shape[0],time.time()-clk, time.time()-local_clk))






                # for cur_prod in f:
                #     cur_prod = eval(cur_prod)
                #     all_cat.append(asin_cat[cur_prod["asin"]])
                #     all_text.append(cur_prod["reviewText"])
                # print("\t%d"%len(all_text), end="\t")
            #     result_text = p.map(func=self.sentences_preprocess, iterable=all_text)
            #     p.close()
            #     p.join()
            # print(time.time()-local_time)

def extract_review():
    review_path = "/media/wltjr1007/nvme/HW/data/review/"
    write_path= "/media/wltjr1007/nvme/HW/data/"

    asin_cat = np.load("/media/wltjr1007/nvme/HW/data/asin_cat.npy").item()
    all_cat = []
    with open(write_path+"extracted_reviews.txt", "w") as ff:
        for file_cnt, file_name in enumerate(listdir(review_path)):
            print(file_cnt, file_name, end="")
            with open(review_path+file_name, "r") as f:
                for cur_prod in f:
                    cur_prod = eval(cur_prod)
                    all_cat.append(asin_cat[cur_prod["asin"]])
                    ff.write(cur_prod["reviewText"]+"\n")
                print("\t%d"%len(all_cat), end="\t")
    np.save(write_path+"extracted_reviews_idx.npy", all_cat)

def extract_document():
    review_path = "/media/wltjr1007/nvme/HW/data/parsed/"
    path = "/media/wltjr1007/nvme/HW/data/document2_review.txt"
    review_filename = [int(f.split(".")[0]) for f in listdir(review_path)]
    review_filename.sort()
    with open(path, "w") as ff:
        for i in review_filename:
            with open(review_path+"%d.txt"%i, "r") as f:
                ff.write(" ".join([line.strip() for line in f][::10])+"\n")
            print(i)


def save_tfidf():
    path = "/media/wltjr1007/nvme/HW/data/document2_review.txt"
    dict_save_path = "/media/wltjr1007/nvme/HW/data/document2_review.dictionary"
    tfidf_save_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"

    with open(path, "r") as f:
        document = [cur_doc.split() for cur_doc in f]
    print("%d Documents"%len(document))
    dictionary = gensim.corpora.Dictionary(document)

    dictionary.save(dict_save_path)
    # dictionary = gensim.corpora.Dictionary.load(path, mmap="r")
    corpus = [dictionary.doc2bow(cur_doc) for cur_doc in document]
    tfidf = gensim.models.TfidfModel(corpus=corpus)
    tfidf.save(tfidf_save_path)

def get_tfidf():
    tfidf_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"

    model = gensim.models.TfidfModel.load(tfidf_load_path)
    print(model[[(1,2)]])


def train_doc2vec():
    from gensim.models import doc2vec
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    path = "/media/wltjr1007/nvme/HW/data/document2_review.txt"
    save_path = "/media/wltjr1007/nvme/HW/data/model/%d.doc2vec"
    sentences = doc2vec.TaggedLineDocument(path)
    model = doc2vec.Doc2Vec(sentences, iter=20, workers=5)
    model.save(save_path%0)






def train_word2vec():
    from gensim.models import word2vec
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence('/media/wltjr1007/nvme/HW/data/document2_review.txt')

    save_path = "/media/wltjr1007/nvme/HW/data/model/%d.word2vec"
    model = word2vec.Word2Vec(sentences=sentences, size=500, alpha=0.025, window=5, sample=0.001, workers=5,
                              min_alpha=0.0001, sg=1, hs=0, negative=20, cbow_mean=1, iter=5, null_word=0,
                              trim_rule=None, sorted_vocab=1, batch_words=10000)

    model.save(save_path%5)


def test_doc2vec(query_word, is_tfidf=False):
    from gensim.models import doc2vec
    tfidf_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"
    dict_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.dictionary"
    load_path = "/media/wltjr1007/nvme/HW/data/model/0.doc2vec"

    tfidf_model = gensim.models.TfidfModel.load(tfidf_load_path)
    dictionary = gensim.corpora.Dictionary.load(dict_load_path, mmap="r")
    model = doc2vec.Doc2Vec.load(load_path)

    word_cos=np.array(model.most_similar(positive = [query_word], topn=50))
    word = word_cos[:,0]
    word_sort = word
    cos = word_cos[:,1].astype(np.float)
    cos_sim = word_cos[:,1].astype(np.float)
    print("Doc2Vec Q=\"%s\""%query_word, end=" ")
    tfidf_out =np.array(tfidf_model[dictionary.doc2bow(word)])
    tfidf_out_sorted = tfidf_out[np.argsort(tfidf_out[:,1])][::-1]
    if is_tfidf:
        print("With TF-IDF", end="")
        cos_sim+= tfidf_out[:,-1]
        sort_idx = np.argsort(cos_sim)[::-1]
        cos_sim = cos_sim[sort_idx]
        word_sort = word[sort_idx]
        tfidf_out = tfidf_out[sort_idx]
    print("\nRank\tScore\t\tWord2Vec\t\tTFIDF\t\t\tWord")
    for cnt, (w, c) in enumerate(zip(word_sort, cos_sim)):
        if cnt>=20:
            break
        temp_rank = np.argwhere(word==w).squeeze()
        temp_val = cos[temp_rank]
        print("%d\t\t%f\t%f(%2d)"%(cnt+1, c,temp_val,temp_rank+1), end="")
        temp_val = tfidf_out[cnt,-1]
        temp_rank = np.argwhere(tfidf_out_sorted[:,1]==temp_val)[0].squeeze()
        if not is_tfidf:
            temp_rank = -1
            temp_val = 0
        print("\t%f(%2d)\t%s"%(temp_val, temp_rank+1, w))

    print()


def test_word2vec(query_word, is_tfidf=False):
    from gensim.models import word2vec
    dict_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.dictionary"
    tfidf_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"
    load_path = "/media/wltjr1007/nvme/HW/data/model/5.word2vec"

    tfidf_model = gensim.models.TfidfModel.load(tfidf_load_path)
    dictionary = gensim.corpora.Dictionary.load(dict_load_path, mmap="r")
    model = word2vec.Word2Vec.load(load_path)
    word_cos=np.array(model.most_similar(positive = [query_word], topn=50))
    word = word_cos[:,0]
    word_sort = word
    cos = word_cos[:,1].astype(np.float)
    cos_sim = word_cos[:,1].astype(np.float)
    print("Word2Vec Q=\"%s\""%query_word, end=" ")
    tfidf_out =np.array(tfidf_model[dictionary.doc2bow(word)])
    tfidf_out_sorted = tfidf_out[np.argsort(tfidf_out[:,1])][::-1]
    if is_tfidf:
        print("With TF-IDF", end="")
        cos_sim+= tfidf_out[:,-1]
        sort_idx = np.argsort(cos_sim)[::-1]
        cos_sim = cos_sim[sort_idx]
        word_sort = word[sort_idx]
        tfidf_out = tfidf_out[sort_idx]
    print("\nRank\tScore\t\tWord2Vec\t\tTFIDF\t\t\tWord")
    for cnt, (w, c) in enumerate(zip(word_sort, cos_sim)):
        if cnt>=20:
            break
        temp_rank = np.argwhere(word==w).squeeze()
        temp_val = cos[temp_rank]
        print("%d\t\t%f\t%f(%2d)"%(cnt+1, c,temp_val,temp_rank+1), end="")
        temp_val = tfidf_out[cnt,-1]
        temp_rank = np.argwhere(tfidf_out_sorted[:,1]==temp_val)[0].squeeze()
        if not is_tfidf:
            temp_rank = -1
            temp_val = 0
        print("\t%f(%2d)\t%s"%(temp_val, temp_rank+1, w))

    print()


def test_ensemble(query_word, is_tfidf=False):
    from gensim.models import doc2vec, word2vec
    tfidf_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"
    dict_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.dictionary"
    doc2vec_load_path = "/media/wltjr1007/nvme/HW/data/model/0.doc2vec"
    word2vec_load_path = "/media/wltjr1007/nvme/HW/data/model/5.word2vec"
    tfidf_model = gensim.models.TfidfModel.load(tfidf_load_path)
    dictionary = gensim.corpora.Dictionary.load(dict_load_path, mmap="r")
    doc2vec_model = doc2vec.Doc2Vec.load(doc2vec_load_path)
    word2vec_model = word2vec.Word2Vec.load(word2vec_load_path)

    doc2vec_word_cos=doc2vec_model.most_similar(positive = [query_word], topn=50)
    word2vec_word_cos=word2vec_model.most_similar(positive = [query_word], topn=50)

    d_word = np.array(doc2vec_word_cos)[:,0]
    d_cos = np.array(doc2vec_word_cos)[:,1]
    w_word = np.array(word2vec_word_cos)[:,0]
    w_cos = np.array(word2vec_word_cos)[:,1]


    all_word_cos = dict(doc2vec_word_cos)

    for wor, cos in word2vec_word_cos:
        if wor not in all_word_cos:
            all_word_cos[wor]=0
        all_word_cos[wor]+=cos

    tfidf_out = tfidf_model[dictionary.doc2bow(all_word_cos.keys())]
    t_word = np.array(tfidf_out)[:,0].astype(np.int)
    t_val = np.array(tfidf_out)[:,1].astype(np.float)
    print("Ensemble Model Q=\"%s\""%query_word, end="")
    if is_tfidf:
        print(" With TF-IDF", end="")
        for word_idx, tfidf in tfidf_out:
            all_word_cos[dictionary.get(word_idx)] += tfidf
    print()
    print("Rank\tScore\t\tDoc2Vec\t\t\tWord2Vec\t\tTFIDF\t\t\tWord")
    for cnt, w in enumerate(sorted(all_word_cos, key=all_word_cos.get, reverse=True)):
        if cnt>=20:
            break
        print("%d\t\t%f\t"%(cnt+1, all_word_cos[w]), end="")
        temp_rank =np.argwhere(d_word == w).squeeze()
        temp_cos = d_cos[temp_rank].astype(np.float)
        if not np.isscalar(temp_cos):
            temp_cos = 0
            temp_rank = -1
        print("%f(%2d)\t" % (temp_cos, temp_rank+1),end="")

        temp_rank = np.argwhere(w_word == w).squeeze()
        temp_cos = w_cos[temp_rank].astype(np.float)
        if not np.isscalar(temp_cos):
            temp_cos = 0
            temp_rank = -1
        print("%f(%2d)\t" % (temp_cos,temp_rank+1), end="")


        temp_rank = np.argwhere(t_word == dictionary.doc2bow([w])[0][0]).squeeze()
        temp_cos = t_val[temp_rank].astype(np.float)
        if not is_tfidf:
            temp_rank = -1
            temp_cos = 0
        print("%f(%2d)\t" % (temp_cos,temp_rank+1), end="")

        print("%s"%w)
    print()

def get_statistic():

    dict_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.dictionary"
    tfidf_load_path = "/media/wltjr1007/nvme/HW/data/document2_review.tfidf"
    dictionary = gensim.corpora.Dictionary.load(dict_load_path, mmap="r")
    tfidf_model = gensim.models.TfidfModel.load(tfidf_load_path)

    print(dictionary)

    print(tfidf_model)



if __name__=="__main__":
    # download()
    # unzip_data()
    # extract_meta()
    # extract_review()
    # pd = preprocess_data()
    # pd.create_data()
    # extract_document()
    # save_tfidf()
    # get_tfidf()
    # train_doc2vec()
    # train_word2vec()

    query_word = "coffee"
    test_doc2vec(is_tfidf=False, query_word=query_word)
    test_doc2vec(is_tfidf=True, query_word=query_word)
    test_word2vec(is_tfidf=False, query_word=query_word)
    test_word2vec(is_tfidf=True, query_word=query_word)
    test_ensemble(is_tfidf=False, query_word=query_word)
    test_ensemble(is_tfidf=True, query_word=query_word)

    # get_statistic()
