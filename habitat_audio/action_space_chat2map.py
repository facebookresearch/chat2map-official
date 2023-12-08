# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions


@registry.register_action_space_configuration(name="chat2map")
class Chat2MapActionSpaceConfiguration(ActionSpaceConfiguration):
    """an extra action (LOOK_UP) is included to accommodate 4 sampling agents for 2 agents without using the STOP action
       value as one of the actions"""
    def get(self):        
        return {

            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "turn_left",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
            HabitatSimActions.LOOK_UP: habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE),
            ),
        }
