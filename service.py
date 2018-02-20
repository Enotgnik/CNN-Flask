
class snService():

    def createINC(sentence, prediction):

        instance = "insance"
        user = "user"
        password = "pass"

        import pysnow

        c = pysnow.Client(instance=instance, user=user, password=password)

        incident = c.resource(api_path="/table/incident")

        new_record = {
            'short_description': sentence,
            'category': prediction
        }

        result = incident.create(payload=new_record)
        print("You incident has been created", result['number'] ,  "Category=",result['category'])

