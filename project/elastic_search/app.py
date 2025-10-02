
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch

app = Flask(__name__)
es = Elasticsearch([{"host": "localhost", "port": 9200}])

@app.route('/search', methods=['POST'])
def semantic_search():
	req = request.get_json()
	query_vector = req.get("vector")
	k = req.get("k", 10)
	# Perform k-NN search on index "motions" using the vector field "motion_vector"
	response = es.search(index="motions", body={
		"size": k,
		"query": {
			"knn": {
				"motion_vector": {
					"vector": query_vector,
					"k": k
				}
			}
		}
	})
	# Return the matching documents (source content) to the client
	hits = [hit["_source"] for hit in response["hits"]["hits"]]
	return jsonify(hits)
