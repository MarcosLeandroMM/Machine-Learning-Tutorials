'''
Em Scikit-Learn, biclustering é uma técnica de agrupamento que vai além do agrupamento regular de linhas ou colunas.  Agrupamento regular agrupa linhas ou colunas independentemente umas das outras.  Biclustering, por outro lado, agrupa simultaneamente linhas e colunas de uma matriz de dados. Esses agrupamentos de linhas e colunas são conhecidos como biclusteres. Cada bicluster identifica uma submatriz da matriz de dados original que compartilha alguma propriedade interessante.

Por exemplo, imagine que você tenha uma matriz onde as linhas são genes e as colunas são amostras de tecido.  O biclustering pode revelar grupos de genes que se comportam de maneira semelhante em grupos específicos de amostras de tecido.

Scikit-Learn oferece dois algoritmos de biclustering:

SpectralCoclustering: Esta técnica baseia-se na ideia de que os dados têm uma estrutura subjacente "em tabuleiro de damas". Ele particiona as linhas e colunas supondo que os dados têm essa estrutura. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html
SpectralBiclustering: Esta técnica pressupõe que os dados têm uma estrutura "em blocos" e tenta encontrar blocos retangulares de dados com comportamentos distintos. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html
É importante ressaltar que atualmente o scikit-learn não oferece métricas internas para avaliar a qualidade dos biclusteres. A avaliação precisa ser feita com métricas externas ao algoritmo.

'''