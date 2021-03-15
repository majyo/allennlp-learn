import tornado.ioloop
import tornado.web
import json
import app.myapp


nlp_app = app.myapp.Application()
args = nlp_app.construct_params_for_predict()
predictor = nlp_app.restore_predictor(args)


class MainHandler(tornado.web.RequestHandler):
    def initialize(self) -> None:
        print("handler init")
        self.nlp_app = nlp_app
        self.predictor = predictor

    def get(self):
        self.write("Hello, world")

    def post(self):
        print(self.request)
        if self.request.headers["Content-Type"] == "application/json":
            self.args = json.loads(self.request.body)
            result = self.nlp_app.predict_json(self.args, predictor)
            self.args["result"] = result
            result = json.dumps(self.args)
            print(self.args)
            self.set_status(200)
            self.set_header("Content-Type", "application/json")
            self.write(result)
            self.flush()
            self.finish()


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("start host")
    tornado.ioloop.IOLoop.current().start()
