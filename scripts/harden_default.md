Also change the silent default in eval/test.cpp (~line 96) so a missing key
is loud rather than silently zero:

    // BEFORE
    system_config.hbm_reserve_fraction =
        config["system"]["hbm_reserve_fraction"].as<double>(0.0);

    // AFTER
    if (!config["system"]["hbm_reserve_fraction"]) {
      std::cerr << "[WARN] hbm_reserve_fraction not set in config; "
                << "defaulting to 0.0. Batch sizes will NOT be comparable "
                << "with calibrated runs.\n";
    }
    system_config.hbm_reserve_fraction =
        config["system"]["hbm_reserve_fraction"].as<double>(0.0);
