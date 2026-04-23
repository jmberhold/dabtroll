.PHONY: check sim g1 groot

check:
	make scripts/sanity_check.sh

sim:
	make scripts/run_isaacsim.sh

g1:
	make scripts/run_g1_task.sh

groot:
	@bash scripts/run_groot_server.sh
