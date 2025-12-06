"""
Zork evaluator for ShinkaEvolve - supports both MIND API and local evaluation
"""

import os
import time
import requests
import tempfile
from typing import Dict, Any, Tuple
from pathlib import Path


def evaluate_zork_local(program_path: str, results_dir: str) -> Dict[str, Any]:
    """
    Evaluate Zork agent locally using Frotz directly (faster, no MIND API queue).
    
    Args:
        program_path: Path to the agent code file
        results_dir: Directory for results
    
    Returns:
        Dict with score and other metrics
    """
    import json
    import subprocess
    import importlib.util
    from pathlib import Path
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    result = {'game_score': 0.0, 'combined_score': 0.0, 'success': False}
    
    try:
        # Import the agent module
        spec = importlib.util.spec_from_file_location("solution", program_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
        
        # Get Zork game file (download if needed)
        import tempfile
        import urllib.request
        zork_url = "https://github.com/BYU-PCCL/z-machine-games/raw/master/jericho-game-suite/zork1.z5"
        
        # Try pre-installed location first
        gamefile = '/usr/local/share/jericho_games/zork1.z5'
        if not os.path.exists(gamefile):
            # Use temp cache
            cache_dir = os.path.join(tempfile.gettempdir(), '.jericho_games')
            os.makedirs(cache_dir, exist_ok=True)
            gamefile = os.path.join(cache_dir, 'zork1.z5')
            
            # Download if not cached
            if not os.path.exists(gamefile):
                print(f"📥 Downloading Zork game file...")
                urllib.request.urlretrieve(zork_url, gamefile)
                print(f"✅ Downloaded to {gamefile}")
        
        # Run 100 episodes locally
        from jericho import FrotzEnv
        env = FrotzEnv(gamefile)
        
        total_score = 0
        num_episodes = 100
        max_steps = 100
        
        print(f"🎮 Running {num_episodes} episodes locally...")
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            
            # Check if solution has ZorkAgent class or predict function
            if hasattr(solution, 'ZorkAgent'):
                agent = solution.ZorkAgent()
                get_action = lambda obs, score, done: agent.act(obs, score, done)
            elif hasattr(solution, 'predict'):
                get_action = lambda obs, score, done: solution.predict(obs)
            else:
                raise AttributeError("Solution must have either ZorkAgent class or predict() function")
            
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                try:
                    action = get_action(obs, info.get('score', 0), done)
                    obs, reward, done, info = env.step(action)
                    steps += 1
                except Exception as e:
                    print(f"Episode {ep+1} error: {e}")
                    break
            
            episode_score = info.get('score', 0)
            total_score += episode_score
            
            if (ep + 1) % 10 == 0:
                avg_so_far = total_score / (ep + 1)
                print(f"  Episodes {ep+1}/{num_episodes}, avg score: {avg_so_far:.2f}")
        
        avg_score = total_score / num_episodes
        print(f"✅ Local evaluation complete: {avg_score:.2f}/350")
        
        # Read code size
        with open(program_path, 'r') as f:
            code = f.read()
        
        result = {
            'game_score': avg_score,
            'combined_score': avg_score,
            'success': True,
            'code_size': len(code),
            'compressed_code_size': len(code),  # No compression in local mode
            'tmp_size': 0,
            'execution_time': 0,
        }
        
    except Exception as e:
        print(f"❌ Local evaluation error: {e}")
        result['error'] = str(e)
        result['success'] = False
    
    # Save metrics.json
    metrics = {
        'combined_score': result['combined_score'],
        'game_score': result['game_score'],
        'code_size': result.get('code_size', 0),
        'compressed_code_size': result.get('compressed_code_size', 0),
        'tmp_size': result.get('tmp_size', 0),
    }
    metrics_file = results_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save correct.json
    correct_data = {'correct': result.get('success', False)}
    correct_file = results_path / 'correct.json'
    with open(correct_file, 'w') as f:
        json.dump(correct_data, f, indent=2)
    
    return result


def evaluate_zork_agent(program_path: str, results_dir: str) -> Dict[str, Any]:
    """
    Main evaluation function called by ShinkaEvolve's eval_hydra.py.
    
    Supports two modes:
    - MIND API mode: Submit to MIND benchmark (official scoring, queue-based)
    - Local mode: Run Frotz directly (faster, no queue wait)
    
    Mode is selected via environment variables:
    - EVAL_MODE: "mind_api" or "local" (default: "mind_api")
    - MIND_API_URL: MIND API endpoint (default: "http://localhost:8002")
    - BENCHMARK_ID: Benchmark ID (default: 4)
    
    Args:
        program_path: Path to the agent code file
        results_dir: Directory for results
    
    Returns:
        Dict with score and other metrics
    """
    eval_mode = os.environ.get("EVAL_MODE", "mind_api").lower()
    
    if eval_mode == "local":
        print(f"🔧 Using LOCAL evaluation mode (Frotz direct)")
        return evaluate_zork_local(program_path, results_dir)
    else:
        print(f"🌐 Using MIND API evaluation mode")
        return evaluate_zork_mind_api(program_path, results_dir)


def evaluate_zork_mind_api(program_path: str, results_dir: str) -> Dict[str, Any]:
    """
    Evaluate using MIND API (queue-based, official scoring).
    
    Args:
        program_path: Path to the agent code file
        results_dir: Directory for results
    
    Returns:
        Dict with score and other metrics
    """
    import json
    from pathlib import Path
    
    API_URL = os.environ.get("MIND_API_URL", "http://localhost:8002")
    BENCHMARK_ID = int(os.environ.get("BENCHMARK_ID", "4"))
    
    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Read the agent code
    with open(program_path, 'r') as f:
        code = f.read()
    
    print(f"📤 Submitting Zork agent to {API_URL}...")
    
    # Submit to MIND API
    files = {
        'solution': ('solution.py', code, 'text/plain')
    }
    data = {
        "benchmark_id": str(BENCHMARK_ID),
        "user_email": "shinka@example.com"
    }
    
    result = {'game_score': 0.0, 'combined_score': 0.0, 'success': False}
    
    try:
        response = requests.post(
            f"{API_URL}/api/submit",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Submission failed: {response.status_code}")
            print(response.text)
        else:
            result_json = response.json()
            submission_id = result_json.get('id')
            
            if submission_id:
                print(f"✅ Submitted as #{submission_id}, waiting for evaluation...")
                
                # Poll for result (max 3 minutes)
                for i in range(90):
                    time.sleep(2)
                    poll_response = requests.get(f"{API_URL}/api/submissions/{submission_id}", timeout=10)
                    
                    if poll_response.status_code == 200:
                        data = poll_response.json()
                        status = data.get('status')
                        
                        if status == 'success':
                            score = data.get('average_ntps', 0.0)
                            exec_time = data.get('execution_time', 0.0)
                            # Use MIND's pre-computed metrics (includes compressed size, tmp_size penalty)
                            api_code_size = data.get('code_size', 0)
                            compressed_size = data.get('compressed_code_size', api_code_size)
                            tmp_size = data.get('tmp_size', 0)
                            print(f"🎮 Score: {score:.1f}/350 (time: {exec_time:.2f}s, code: {compressed_size}b, tmp: {tmp_size}b)")
                            
                            result = {
                                'game_score': float(score),
                                'combined_score': float(score),
                                'execution_time': float(exec_time),
                                'code_size': int(api_code_size),
                                'compressed_code_size': int(compressed_size),
                                'tmp_size': int(tmp_size),
                                'submission_id': submission_id,
                                'success': True
                            }
                            break
                        elif status in ['failed_timeout', 'failed_error']:
                            print(f"❌ Evaluation failed: {status}")
                            break
                    
                    if (i + 1) % 15 == 0:
                        print(f"⏳ Still waiting... ({(i+1)*2}s)")
                
                if not result['success'] and status not in ['failed_timeout', 'failed_error']:
                    print("⏱️ Timeout waiting for result")
                    result['timeout'] = True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        result['error'] = str(e)
    
    # Save results for ShinkaEvolve
    # ShinkaEvolve expects: metrics.json and correct.json as separate files
    
    # Use MIND API's pre-computed code_size (includes compression analysis)
    # Fallback to len(code) only if API didn't provide it
    api_code_size = result.get('code_size', len(code))
    compressed_size = result.get('compressed_code_size', api_code_size)
    tmp_size = result.get('tmp_size', 0)
    
    # Save metrics.json
    metrics = {
        'combined_score': result['combined_score'],
        'game_score': result['game_score'],
        'code_size': api_code_size,
        'compressed_code_size': compressed_size,
        'tmp_size': tmp_size,
    }
    if 'execution_time' in result:
        metrics['execution_time'] = result['execution_time']
    if 'submission_id' in result:
        metrics['submission_id'] = result['submission_id']
    
    metrics_file = results_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save correct.json  
    correct_data = {
        'correct': result.get('success', False)
    }
    correct_file = results_path / 'correct.json'
    with open(correct_file, 'w') as f:
        json.dump(correct_data, f, indent=2)
    
    return result


# Old queue-aware code below (not used by ShinkaEvolve's eval_hydra.py)

def check_queue_status(api_url: str = "http://localhost:8002", benchmark_id: int = 4) -> Dict[str, int]:
    """
    Check the current queue status for pending jobs.
    
    Returns:
        Dict with 'pending', 'running', 'total' counts
    """
    try:
        # Check Redis queue or database for pending jobs
        # This is a simple approach - checking recent submissions
        response = requests.get(
            f"{api_url}/api/leaderboard?benchmark_id={benchmark_id}&limit=100",
            timeout=5
        )
        if response.status_code != 200:
            return {'pending': 0, 'running': 0, 'total': 0}
        
        entries = response.json()['entries']
        
        # Count by status
        pending = sum(1 for e in entries if e.get('status') == 'queued')
        running = sum(1 for e in entries if e.get('status') == 'running')
        
        return {
            'pending': pending,
            'running': running,
            'total': pending + running
        }
    except Exception as e:
        print(f"Warning: Could not check queue status: {e}")
        return {'pending': 0, 'running': 0, 'total': 0}


def submit_solution(code: str, api_url: str, benchmark_id: int) -> Tuple[bool, int, str]:
    """
    Submit a solution to the MIND API.
    
    Returns:
        (success, submission_id, error_message)
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        with open(tmp_path, 'rb') as f:
            response = requests.post(
                f"{api_url}/api/submit",
                files={'solution': ('solution.py', f, 'text/x-python')},
                data={'benchmark_id': str(benchmark_id)},
                timeout=30
            )
        
        os.unlink(tmp_path)
        
        if response.status_code != 200:
            return False, -1, f"Submission failed: {response.text}"
        
        submission_id = response.json()['id']
        return True, submission_id, ""
        
    except Exception as e:
        return False, -1, str(e)


def wait_for_result(submission_id: int, api_url: str, max_wait: int = 180) -> Dict[str, Any]:
    """
    Poll for evaluation results.
    
    Returns:
        Result dictionary with score and metadata
    """
    start_time = time.time()
    poll_interval = 2
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{api_url}/api/submissions/{submission_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                time.sleep(poll_interval)
                continue
            
            submission = response.json()
            status = submission['status']
            
            if status in ['success', 'completed']:
                return {
                    'success': True,
                    'game_score': float(submission.get('average_ntps', 0.0)),
                    'execution_time': submission.get('execution_time', 0.0),
                    'submission_id': submission_id,
                    'status': status
                }
            elif status in ['failed_error', 'failed_invalid', 'failed_timeout']:
                return {
                    'success': False,
                    'game_score': 0.0,
                    'error': submission.get('error_message', status),
                    'submission_id': submission_id,
                    'status': status
                }
            
            # Still running
            time.sleep(poll_interval)
            
        except Exception as e:
            time.sleep(poll_interval)
    
    return {
        'success': False,
        'game_score': 0.0,
        'error': 'Timeout waiting for results',
        'submission_id': submission_id,
        'status': 'timeout'
    }


def evaluate_with_queue_awareness(
    program_path: str,
    api_url: str = "http://localhost:8002",
    benchmark_id: int = 4,
    max_queue_depth: int = 5
) -> Dict[str, Any]:
    """
    Evaluate a Zork agent with queue-aware submission.
    
    Args:
        program_path: Path to the Python solution
        api_url: MIND Challenge API URL
        benchmark_id: Benchmark ID (4 for Zork)
        max_queue_depth: Maximum queue depth to wait for
        
    Returns:
        Evaluation results
    """
    # Read solution code
    with open(program_path, 'r') as f:
        code = f.read()
    
    # Check queue status
    queue_status = check_queue_status(api_url, benchmark_id)
    pending = queue_status['pending']
    running = queue_status['running']
    
    print(f"📊 Queue status: {pending} pending, {running} running")
    
    # Adaptive waiting: if queue is too deep, wait before submitting
    if pending > max_queue_depth:
        wait_time = min(30, pending * 2)  # Wait up to 30 seconds
        print(f"⏳ Queue depth {pending} > {max_queue_depth}, waiting {wait_time}s...")
        time.sleep(wait_time)
    
    # Submit solution
    print(f"📤 Submitting solution...")
    success, submission_id, error = submit_solution(code, api_url, benchmark_id)
    
    if not success:
        return {
            'game_score': 0.0,
            'combined_score': 0.0,
            'success': False,
            'error': error
        }
    
    print(f"✅ Submitted as #{submission_id}, waiting for results...")
    
    # Wait for results
    result = wait_for_result(submission_id, api_url)
    
    if result['success']:
        score = result['game_score']
        print(f"🎮 Score: {score}/350")
        return {
            'game_score': score,
            'combined_score': score,  # ShinkaEvolve uses this for optimization
            'execution_time': result['execution_time'],
            'submission_id': submission_id,
            'success': True
        }
    else:
        print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        return {
            'game_score': 0.0,
            'combined_score': 0.0,
            'success': False,
            'error': result.get('error', 'Unknown error'),
            'submission_id': submission_id
        }


def main(program_path: str, results_dir: str, **kwargs):
    """
    Main evaluation entry point for ShinkaEvolve.
    
    This function is called by the ShinkaEvolve framework.
    """
    api_url = kwargs.get('api_url', os.getenv('MIND_API_URL', 'http://localhost:8002'))
    benchmark_id = kwargs.get('benchmark_id', 4)
    max_queue_depth = kwargs.get('max_queue_depth', 5)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {program_path}")
    print(f"API: {api_url}")
    print(f"{'='*60}\n")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Run evaluation with queue awareness
    result = evaluate_with_queue_awareness(
        program_path=program_path,
        api_url=api_url,
        benchmark_id=benchmark_id,
        max_queue_depth=max_queue_depth
    )
    
    # Save results
    result_file = os.path.join(results_dir, 'result.txt')
    with open(result_file, 'w') as f:
        f.write(f"Score: {result.get('game_score', 0)}/350\n")
        f.write(f"Success: {result.get('success', False)}\n")
        if 'submission_id' in result:
            f.write(f"Submission ID: {result['submission_id']}\n")
        if 'error' in result:
            f.write(f"Error: {result['error']}\n")
    
    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <program_path> [results_dir]")
        sys.exit(1)
    
    program_path = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) > 2 else './results'
    
    result = main(program_path, results_dir)
    print(f"\n{'='*60}")
    print(f"Final Result: {result}")
    print(f"{'='*60}")
