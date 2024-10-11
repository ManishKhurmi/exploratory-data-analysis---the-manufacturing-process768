strategy_logit_machine_failure_all_product_types = {'torque': [60], 
            #  'rotational_speed_actual': [], # No value as the probability of machine failure is < 5% for the given data 
             'air_temperature': [304], 
            #  'process_temperature': , # No value as the probability of machine failure is < 5% for the given data 
             'tool_wear': [240]}
print('\nStrategy of Logit Model (y=`machine_failure) Across ALL Prouct Types')
print(strategy_logit_machine_failure_all_product_types)