#convertir le fichier .bin au fichier .txt 
from gensim.models import KeyedVectors 
path = './embeddings/frWac_no_postag_no_phrase_700_skip_cut50.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True)  
model.save_word2vec_format('frWac_no_postag_no_phrase_700_skip_cut50.txt', binary=False)



################# chargement des embeddings et indices des mots#################
class Indices:
	""" class to handle the correspondances from indices to words 
	and from words to indices correspondances """

	def __init__(self):
		self.id2w = []
		self.w2id = {}

	def get_nb_words(self):
		return len(self.id2w)

	def id_from_w(self, w):
		""" 
		returns index of word w if w is not already known,
		returns None if create_new is False,
		otherwise, creates a new index for w and returns it 
		"""
		if w in self.w2id:
			return self.w2id[w]
		#if not create_new:
			#return None
		self.w2id[w] = len(self.id2w)
		#@print("WORD %s associated to id %i" % (w, len(self.i2w)))
		self.id2w.append(w)

		return self.w2id[w]

def read_examples(infile, indices):
	"""
	lit un fichier .txt de vecteurs de mots et retourne la matrice de vecteur et 
	met à jour les indices
	
	"""
	with open(infile) as stream:
	# récupère nb de mot et taille de vec dans la 1ère ligne
	# puis initalise une matrice de shape (nb_mot, taille_vec) 
		line = stream.readline()
		line1 = line.strip('\n').split(" ")
		nb_mot = int(line1[0])
		taille_vec = int(line1[1])
		matrice = np.zeros((nb_mot,taille_vec))
		
		# lit la ligne suivante
		line = stream.readline()
		# tant que ligne n'est pas vide
		while line:
			# transforme le vecteur du mot en une liste de string
			line = line.strip('\n').split(" ")
			# mot courant est le premier élément de liste
			mot = line[0]
			# indice du mot courant
			id = indices.id_from_w(mot)
			# met à jour la matrice: une ligne stock un veteur de mot
			for i in range(1,taille_vec+1):
				row = id
				colomn = i-1
				matrice[row,colomn] = float(line[i])
			#lit la linge suivante
			line = stream.readline()
		
		return matrice
    

indices = Indices()
matrice = read_examples(repertoire_vec_mots,indices)
