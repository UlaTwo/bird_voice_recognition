.PHONY: dashboard train

dashboard:
	@cd dashboard && streamlit run main.py --browser.serverAddress 127.0.0.1

train:
	@python src/train.py -ag True -n model -d 1 -e 1 -w True -v False

warblr:
	@python src/check_warblr.py -m T+B+F_model_25e_vallist.ckpt