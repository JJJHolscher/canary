
all:  .venv .venv/lib/python3.11/site-packages/mods
	. .venv/bin/activate && \
	python -m minetester.scripts.test_loop

.venv: minetest/build/package/wheel/minetester-0.0.1-py3-none-any.whl requirements.txt
	python -m venv --prompt canary .venv
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

.venv/lib/python3.11/site-packages/mods: .venv minetest/mods
	mkdir -p .venv/lib/python3.11/site-packages/minetest
	ln -s minetest/mods .venv/lib/python3.11/site-packages/minetest/mods
	ln -s minetest/mods .venv/lib/python3.11/site-packages/mods
	ln -s minetest/clientmods .venv/lib/python3.11/site-packages/minetest/clientmods
	ln -s minetest/clientmods .venv/lib/python3.11/site-packages/clientmods

minetest/build/package/wheel/minetester-0.0.1-py3-none-any.whl:
	echo "build https://github.com/EleutherAI/minetest first"

clean:
	rm -r .venv
