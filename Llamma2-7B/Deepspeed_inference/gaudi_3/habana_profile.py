import torch
from habana_frameworks.torch.activity_profiler import DebugActivity


class HabanaProfile(object):

    HABANA_PROFILE_ENABLED = True

    def __init__(self, warmup: int = 0, active: int = 0, output_dir: str = "./hpu_profile", wait: int = 0):
        if active <= 0 or warmup <= 0 or not HabanaProfile.HABANA_PROFILE_ENABLED:

            def noop():
                pass

            self.start = noop
            self.stop = noop
            self.step = noop
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = False
            schedule = torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1)
            activities = [torch.profiler.ProfilerActivity.CPU,
                          torch.profiler.ProfilerActivity.HPU]

            profiler = torch.profiler.profile(
                schedule=schedule,
                activities=activities,
                debug_activities=[
                    DebugActivity.SYNAPSE_FUNCTION_CALLS, DebugActivity.BRIDGE_FUNCTION_CALLS],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_dir),
                record_shapes=True,
                with_stack=True,
            )
            self.start = profiler.start
            self.stop = profiler.stop
            self.step = profiler.step
            HabanaProfile.enable.invalid = True
            HabanaProfile.disable.invalid = True

    def stop(self):
        self.stop()

    def start(self):
        self.start()

    def step(self):
        self.step()

    @staticmethod
    def disable():
        """
        Runs only once and must happen before doing profiling.
        """
        if hasattr(HabanaProfile.disable, "invalid"):
            if not HabanaProfile.disable.invalid:
                HabanaProfile.HABANA_PROFILE_ENABLED = False
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = False

    @staticmethod
    def enable():
        """
        Runs only once and must happen before doing profiling.
        """
        if hasattr(HabanaProfile.enable, "invalid"):
            if not HabanaProfile.enable.invalid:
                HabanaProfile.HABANA_PROFILE_ENABLED = True
        else:
            HabanaProfile.HABANA_PROFILE_ENABLED = True

