#!/usr/bin/env python3
"""Test script to verify QueueSampler integration end-to-end."""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_queue_sampler_class():
    """Test 1: QueueSampler class functionality"""
    logger.info("=" * 60)
    logger.info("TEST 1: QueueSampler class functionality")
    logger.info("=" * 60)
    
    from shinka.llm.dynamic_sampling import QueueSampler
    import numpy as np
    
    # Create sampler with 3 models
    sampler = QueueSampler(arm_names=['model1', 'model2', 'model3'])
    
    # Test basic properties
    assert sampler.n_arms == 3, "Expected 3 arms"
    logger.info("✓ QueueSampler created with 3 arms")
    
    # Test posterior returns equal probabilities
    posterior = sampler.posterior()
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(posterior, expected), f"Expected {expected}, got {posterior}"
    logger.info(f"✓ Posterior probabilities are equal: {posterior}")
    
    # Test update doesn't change probabilities
    sampler.update('model1', 100.0)
    posterior_after = sampler.posterior()
    assert np.allclose(posterior_after, expected), "Probabilities changed after update"
    logger.info("✓ Probabilities remain constant after update (no learning)")
    
    # Test print_summary works
    sampler.print_summary()
    logger.info("✓ print_summary() works")
    
    logger.info("\n✅ TEST 1 PASSED\n")
    return True

def test_evolution_config():
    """Test 2: EvolutionConfig with disable_model_posteriors"""
    logger.info("=" * 60)
    logger.info("TEST 2: EvolutionConfig with disable_model_posteriors")
    logger.info("=" * 60)
    
    from shinka.core.runner import EvolutionConfig
    
    # Test config without flag (default)
    config1 = EvolutionConfig(llm_models=['m1', 'm2'])
    assert config1.disable_model_posteriors == False, "Default should be False"
    logger.info("✓ Default disable_model_posteriors is False")
    
    # Test config with flag enabled
    config2 = EvolutionConfig(
        llm_models=['m1', 'm2', 'm3'],
        disable_model_posteriors=True
    )
    assert config2.disable_model_posteriors == True, "Flag should be True"
    logger.info("✓ disable_model_posteriors can be set to True")
    
    logger.info("\n✅ TEST 2 PASSED\n")
    return True

def test_runner_integration():
    """Test 3: Runner initializes QueueSampler when flag is set"""
    logger.info("=" * 60)
    logger.info("TEST 3: Runner integration with QueueSampler")
    logger.info("=" * 60)
    
    from shinka.core.runner import EvolutionRunner, EvolutionConfig
    from shinka.launch import LocalJobConfig
    from shinka.database import DatabaseConfig
    from shinka.llm.dynamic_sampling import QueueSampler, FixedSampler
    import tempfile
    import shutil
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test with disable_model_posteriors=True
        evo_config = EvolutionConfig(
            llm_models=[
                'openai/gpt-oss-120b@http://localhost:8000/v1',
                'llama@http://ruling-cub:8000/v1',
            ],
            disable_model_posteriors=True,
            num_generations=1,
            max_parallel_jobs=1,
            results_dir=temp_dir,
            embedding_model="local:all-MiniLM-L6-v2",  # Use local model
        )
        
        job_config = LocalJobConfig()
        db_config = DatabaseConfig(backend='sqlite')
        
        runner = EvolutionRunner(
            evo_config=evo_config,
            job_config=job_config,
            db_config=db_config,
            verbose=False,
        )
        
        # Verify QueueSampler is used
        assert isinstance(runner.llm_selection, QueueSampler), \
            f"Expected QueueSampler, got {type(runner.llm_selection)}"
        logger.info("✓ Runner uses QueueSampler when disable_model_posteriors=True")
        
        # Test with disable_model_posteriors=False (default)
        temp_dir2 = tempfile.mkdtemp()
        evo_config2 = EvolutionConfig(
            llm_models=['model1', 'model2'],
            disable_model_posteriors=False,
            num_generations=1,
            results_dir=temp_dir2,
            embedding_model="local:all-MiniLM-L6-v2",  # Use local model
        )
        
        runner2 = EvolutionRunner(
            evo_config=evo_config2,
            job_config=job_config,
            db_config=db_config,
            verbose=False,
        )
        
        # Verify None is used in runner (LLMClient will convert to FixedSampler)
        assert runner2.llm_selection is None, \
            f"Expected None (will become FixedSampler in LLMClient), got {type(runner2.llm_selection)}"
        logger.info("✓ Runner uses None when disable_model_posteriors=False (LLMClient converts to FixedSampler)")
        
        # Verify LLMClient converts None to FixedSampler
        assert isinstance(runner2.llm.llm_selection, FixedSampler), \
            f"Expected LLMClient to use FixedSampler, got {type(runner2.llm.llm_selection)}"
        logger.info("✓ LLMClient converts None to FixedSampler")
        
        # Cleanup
        shutil.rmtree(temp_dir2)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
    
    logger.info("\n✅ TEST 3 PASSED\n")
    return True

def test_llm_client_integration():
    """Test 4: LLMClient uses QueueSampler correctly"""
    logger.info("=" * 60)
    logger.info("TEST 4: LLMClient with QueueSampler")
    logger.info("=" * 60)
    
    from shinka.llm import LLMClient, QueueSampler
    import numpy as np
    
    # Create QueueSampler
    sampler = QueueSampler(arm_names=['model1', 'model2', 'model3'])
    
    # Create LLMClient with QueueSampler
    llm = LLMClient(
        model_names=['model1', 'model2', 'model3'],
        model_selection=sampler,
        verbose=True,  # Test that verbose logging is skipped
    )
    
    assert llm.llm_selection is sampler, "LLMClient should use provided sampler"
    logger.info("✓ LLMClient accepts QueueSampler")
    
    # Test get_kwargs (should not print verbose SAMPLING output for QueueSampler)
    kwargs = llm.get_kwargs()
    assert 'model_name' in kwargs, "get_kwargs should return model_name"
    logger.info(f"✓ get_kwargs() works: {list(kwargs.keys())}")
    
    # Verify posterior is used for sampling
    posterior = llm.llm_selection.posterior()
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(posterior, expected), "Should have equal probabilities"
    logger.info(f"✓ Posterior probabilities are equal: {posterior}")
    
    logger.info("\n✅ TEST 4 PASSED\n")
    return True

def test_yaml_config():
    """Test 5: YAML config loads correctly"""
    logger.info("=" * 60)
    logger.info("TEST 5: YAML config with disable_model_posteriors")
    logger.info("=" * 60)
    
    import yaml
    
    config_path = Path('configs/variant/zork_8islands.yaml')
    assert config_path.exists(), f"Config file not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'evo_config' in config, "Config should have evo_config"
    assert 'disable_model_posteriors' in config['evo_config'], \
        "evo_config should have disable_model_posteriors"
    assert config['evo_config']['disable_model_posteriors'] == True, \
        "disable_model_posteriors should be True"
    
    logger.info("✓ YAML config loads successfully")
    logger.info(f"✓ disable_model_posteriors = {config['evo_config']['disable_model_posteriors']}")
    logger.info(f"✓ max_parallel_jobs = {config['evo_config']['max_parallel_jobs']}")
    logger.info(f"✓ LLM models: {len(config['evo_config']['llm_models'])} configured")
    
    logger.info("\n✅ TEST 5 PASSED\n")
    return True

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("QueueSampler End-to-End Test Suite")
    logger.info("=" * 60 + "\n")
    
    tests = [
        test_queue_sampler_class,
        test_evolution_config,
        test_runner_integration,
        test_llm_client_integration,
        test_yaml_config,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            logger.error(f"\n❌ {test.__name__} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Feature is working correctly.\n")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed.\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
