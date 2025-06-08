import pickle
tf=pickle.load(open('vectorizer.pkl','rb')) 
# model = pickle.load(open('voting.pkl','rb'))

# txt=['account', 'click','claim' ,'congratul','gift','now','voucher','win','won']
# vector_input = tf.transform(txt).toarray()
# print(vector_input)
# print(model.predict(vector_input))

from main import transform_text
txt=transform_text("Hey, are we still meeting at the library at 4 PM to study for the math test?")

print(txt)