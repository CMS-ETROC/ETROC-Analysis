#############################################################################
# zlib License
#
# (C) 2025 Cristóvão Beirão da Cruz e Silva <cbeiraod@cern.ch>
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#############################################################################


# This script implements a few different functions to control the iseg power supply used at IRRAD through the isegControl2 GUI interface

# Load this script from the python terminal of the GUI with the following command:
#   exec(open([path_to_this_script], "r").read())
#   exec(open("/home/daq/ETROC2_KCU105/ETROC-Analysis/iseg_control.py", "r").read())
# Then run your desired function

connection = icsClient.getConnections()[0]
module = connection.getModules()[0]
channels = module.channels

def wait_ramping():
    import time
    global channels

    for ch in channels:
        while ch.getStatusRamping():
            time.sleep(0.1)

def wait_accuracy(set_value, wait_accuracy):
    import time
    global channels

    for ch in channels:
        while abs(abs(ch.getControlVoltageSet()) - abs(set_value)) > wait_accuracy:
            time.sleep(0.1)

def all_compliance():
    global channels

    for ch in channels:
        if not ch.getStatusConstantCurrent():
            return False

    return True

def is_sorted_ascending(lst):
    """Return Trues if the list is in ascending order, else False"""
    return all(lst[i] < lst [i+1] for i in range(len(lst) - 1))

def quick_iv(
        voltage_step = 10, 
        start_voltage = 0,
        idle_voltage = 0,
        initial_wait = 10,
        step_wait = 10,
        delayed_start = None,
        voltage_accuracy = 0.1,
        ):
    import time
    from datetime import datetime

    global channels
    global wait_ramping
    global all_compliance
    global wait_accuracy
    
    if start_voltage >= 550:
        print("You have set the start voltage too high. Assuming 0V")
        start_voltage = 0
    
    if idle_voltage >= 550:
        print("You have set the idle voltage too high. Assuming 0V")
        idle_voltage = 0

    if delayed_start is not None and delayed_start > 0:
        time.sleep(delayed_start)

    # Set all channels to start value
    for ch in channels:
        ch.setControlVoltageSet(start_voltage)
    wait_ramping()
    if voltage_accuracy is not None and voltage_accuracy > 0:
        wait_accuracy(start_voltage, voltage_accuracy)
    time.sleep(initial_wait)

    # Do the Scan
    print("Starting scan at:", datetime.now())
    current_voltage = start_voltage
    while(True):
        current_voltage += voltage_step
        if current_voltage >= 550:
            break
        for ch in channels:
            ch.setControlVoltageSet(current_voltage)
        wait_ramping()
        if voltage_accuracy is not None and voltage_accuracy > 0:
            wait_accuracy(current_voltage, voltage_accuracy)
        time.sleep(step_wait)

        if all_compliance():
            break
    print("Finished scan at:", datetime.now())

    # Set all channels to idle value since we are done
    for ch in channels:
        ch.setControlVoltageSet(idle_voltage)
    wait_ramping()
    if voltage_accuracy is not None and voltage_accuracy > 0:
        wait_accuracy(idle_voltage, voltage_accuracy)

#fine_iv(idle_voltage=100,step_wait=10,start_voltage=8)
#fine_iv(idle_voltage=100,step_wait=10,start_voltage=8, step_sizes = [0.1, 0.2, ....])
def fine_iv(
        start_voltage = 0,
        idle_voltage = 0,
        initial_wait = 10,
        step_wait = 10,
        step_sizes          = [1, 0.1, 0.1, 0.2,  1,   5, 10],
        step_voltage_limits = [10, 20,  60,  70, 80, 130],
        extra_wait = 10,
        extra_wait_after = 210,
        delayed_start = None,
        voltage_accuracy = 0.1,
        ):
    import time
    import bisect
    from datetime import datetime

    global channels
    global wait_ramping
    global all_compliance
    global is_sorted_ascending
    global wait_accuracy

    # Check some minimum requirements are met for the function to work
    if len(step_sizes) - 1 != len(step_voltage_limits):
        print("The step_sizes list should have 1 more element than the step_voltage_limits list")
        return
    
    if not is_sorted_ascending(step_voltage_limits):
        print("The voltage limits must be a list in ascending order to define the regions where each step size is applied")
        return
    
    if start_voltage >= 550:
        print("You have set the start voltage too high. Assuming 0V")
        start_voltage = 0
    
    if idle_voltage >= 550:
        print("You have set the idle voltage too high. Assuming 0V")
        idle_voltage = 0

    if delayed_start is not None and delayed_start > 0:
        time.sleep(delayed_start)

    # Set all channels to start value
    for ch in channels:
        ch.setControlVoltageSet(start_voltage)
    wait_ramping()
    if voltage_accuracy is not None and voltage_accuracy > 0:
        wait_accuracy(start_voltage, voltage_accuracy)
    time.sleep(initial_wait)

    # Do the Scan
    print("Starting scan at:", datetime.now())
    current_voltage = start_voltage
    while(True):
        idx = bisect.bisect_right(step_voltage_limits, current_voltage)
        voltage_step = step_sizes[idx]
        current_voltage += voltage_step
        if current_voltage >= 550:
            break
        for ch in channels:
            ch.setControlVoltageSet(current_voltage)
        wait_ramping()
        if voltage_accuracy is not None and voltage_accuracy > 0:
            wait_accuracy(current_voltage, voltage_accuracy)
        time.sleep(step_wait)
        if current_voltage >= extra_wait_after:
            time.sleep(extra_wait)

        if all_compliance():
            break
    print("Finished scan at:", datetime.now())

    # Set all channels to idle value since we are done
    for ch in channels:
        ch.setControlVoltageSet(idle_voltage)
    wait_ramping()
    if voltage_accuracy is not None and voltage_accuracy > 0:
        wait_accuracy(idle_voltage, voltage_accuracy)


# repeated_fine_iv(delayed_start=60*60*6, max_repeats=3)
def repeated_fine_iv(
        wait_between_scans = 60*60*2, # 2 Hours between scans
        start_voltage = 0,
        idle_voltage = 0,
        initial_wait = 10,
        step_wait = 10,
        step_sizes          = [1, 0.1, 0.1, 0.2,  1,   5, 10],
        step_voltage_limits = [10, 20,  60,  70, 80, 130],
        extra_wait = 10,
        extra_wait_after = 210,
        max_repeats = None,
        delayed_start = None,
        voltage_accuracy = 0.1,
        ):
    import time

    global fine_iv

    if delayed_start is not None and delayed_start > 0:
        time.sleep(delayed_start)

    counter = 0
    while True:
        fine_iv(
            start_voltage=start_voltage,
            idle_voltage=idle_voltage,
            initial_wait=initial_wait,
            step_wait=step_wait,
            step_sizes=step_sizes,
            step_voltage_limits=step_voltage_limits,
            extra_wait=extra_wait,
            extra_wait_after=extra_wait_after,
            delayed_start=None,
            voltage_accuracy=voltage_accuracy,
            )
        counter += 1
        if max_repeats is not None and counter == max_repeats:
            break
        time.sleep(wait_between_scans)