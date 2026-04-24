#include "Tile.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(__has_include)
#if __has_include(<tensorflow/c/c_api.h>)
#include <tensorflow/c/c_api.h>
#define THE_FLOOR_HAS_TF_C_API 1
#endif
#endif

namespace
{
	constexpr const char* kSavedModelPath = "model/floor_ai_savedmodel";
	constexpr const char* kNormPath = "model/floor_ai.norm.json";

	bool PathExists(const char* filePath)
	{
		struct stat info;
		return stat(filePath, &info) == 0;
	}

	std::vector<float> ParseJsonArray(const std::string& content, const std::string& key)
	{
		std::vector<float> values;
		const std::size_t keyPos = content.find("\"" + key + "\"");
		if (keyPos == std::string::npos)
		{
			return values;
		}

		const std::size_t arrayStart = content.find('[', keyPos);
		const std::size_t arrayEnd = content.find(']', arrayStart);
		if (arrayStart == std::string::npos || arrayEnd == std::string::npos || arrayEnd <= arrayStart)
		{
			return values;
		}

		std::string payload = content.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
		for (char& ch : payload)
		{
			if (ch == '\n' || ch == '\r' || ch == '\t')
			{
				ch = ' ';
			}
		}

		std::stringstream parser(payload);
		std::string token;
		while (std::getline(parser, token, ','))
		{
			std::stringstream numberParser(token);
			float value = 0.0f;
			numberParser >> value;
			if (!numberParser.fail())
			{
				values.push_back(value);
			}
		}

		return values;
	}

#ifdef THE_FLOOR_HAS_TF_C_API
	class TensorFlowPredictor
	{
	public:
		static TensorFlowPredictor& Get()
		{
			static TensorFlowPredictor predictor;
			return predictor;
		}

		int Predict(const std::array<float, FLAT_STATE_SIZE>& flatState, int validNeighborCount)
		{
			if (!EnsureLoaded())
			{
				return -1;
			}

			std::array<float, FLAT_STATE_SIZE> normalized = flatState;
			if (normMean.size() == FLAT_STATE_SIZE && normStd.size() == FLAT_STATE_SIZE)
			{
				for (int i = 0; i < FLAT_STATE_SIZE; ++i)
				{
					normalized[i] = (normalized[i] - normMean[i]) / normStd[i];
				}
			}

			const int64_t dims[2] = { 1, FLAT_STATE_SIZE };
			TF_Tensor* inputTensor = TF_AllocateTensor(TF_FLOAT, dims, 2, sizeof(float) * FLAT_STATE_SIZE);
			if (inputTensor == nullptr)
			{
				return -1;
			}

			std::memcpy(TF_TensorData(inputTensor), normalized.data(), sizeof(float) * FLAT_STATE_SIZE);

			TF_Output input{ inputOp, 0 };
			TF_Output output{ outputOp, 0 };
			TF_Tensor* outputTensor = nullptr;
			TF_SessionRun(
				session,
				nullptr,
				&input,
				&inputTensor,
				1,
				&output,
				&outputTensor,
				1,
				nullptr,
				0,
				nullptr,
				status);

			TF_DeleteTensor(inputTensor);

			if (TF_GetCode(status) != TF_OK || outputTensor == nullptr)
			{
				return -1;
			}

			const auto* qValues = static_cast<const float*>(TF_TensorData(outputTensor));
			const std::int64_t count = TF_TensorByteSize(outputTensor) / static_cast<std::int64_t>(sizeof(float));
			const int boundedCount = std::max(1, std::min(validNeighborCount, static_cast<int>(count)));

			int bestIndex = 0;
			float bestValue = qValues[0];
			for (int i = 1; i < boundedCount; ++i)
			{
				if (qValues[i] > bestValue)
				{
					bestValue = qValues[i];
					bestIndex = i;
				}
			}

			TF_DeleteTensor(outputTensor);
			return bestIndex;
		}

		bool IsLoaded() const
		{
			return session != nullptr;
		}

		~TensorFlowPredictor()
		{
			if (session != nullptr)
			{
				TF_CloseSession(session, status);
				TF_DeleteSession(session, status);
			}

			if (graph != nullptr)
			{
				TF_DeleteGraph(graph);
			}

			if (sessionOptions != nullptr)
			{
				TF_DeleteSessionOptions(sessionOptions);
			}

			if (status != nullptr)
			{
				TF_DeleteStatus(status);
			}
		}

	private:
		TF_Graph* graph = nullptr;
		TF_Status* status = nullptr;
		TF_SessionOptions* sessionOptions = nullptr;
		TF_Session* session = nullptr;
		TF_Operation* inputOp = nullptr;
		TF_Operation* outputOp = nullptr;
		std::vector<float> normMean;
		std::vector<float> normStd;

		TensorFlowPredictor() = default;

		TF_Operation* FirstExistingOp(const std::vector<const char*>& names)
		{
			for (const char* name : names)
			{
				TF_Operation* op = TF_GraphOperationByName(graph, name);
				if (op != nullptr)
				{
					return op;
				}
			}

			return nullptr;
		}

		bool EnsureLoaded()
		{
			if (session != nullptr)
			{
				return true;
			}

			if (!PathExists(kSavedModelPath))
			{
				return false;
			}

			graph = TF_NewGraph();
			status = TF_NewStatus();
			sessionOptions = TF_NewSessionOptions();
			const char* tags = "serve";

			session = TF_LoadSessionFromSavedModel(
				sessionOptions,
				nullptr,
				kSavedModelPath,
				&tags,
				1,
				graph,
				nullptr,
				status);

			if (TF_GetCode(status) != TF_OK || session == nullptr)
			{
				std::cerr << "[TensorFlowPredictor] Failed to load SavedModel from " << kSavedModelPath << "\n";
				return false;
			}

			inputOp = FirstExistingOp({
				"serving_default_keras_tensor",
				"serving_default_input_1",
				"serving_default_inputs",
				"serving_default_dense_input",
				"serving_default_input_layer"
				});
			outputOp = FirstExistingOp({ "StatefulPartitionedCall", "PartitionedCall", "Identity" });

			if (inputOp == nullptr || outputOp == nullptr)
			{
				std::cerr << "[TensorFlowPredictor] Could not resolve SavedModel input/output ops.\n";
				return false;
			}

			LoadNormalization();
			return true;
		}

		void LoadNormalization()
		{
			if (!PathExists(kNormPath))
			{
				return;
			}

			std::ifstream normFile(kNormPath);
			if (!normFile.is_open())
			{
				return;
			}

			std::stringstream buffer;
			buffer << normFile.rdbuf();
			const std::string content = buffer.str();
			normMean = ParseJsonArray(content, "feature_mean");
			normStd = ParseJsonArray(content, "feature_std");

			if (normMean.size() != FLAT_STATE_SIZE || normStd.size() != FLAT_STATE_SIZE)
			{
				normMean.clear();
				normStd.clear();
			}
		}
	};
#endif

	int PredictActionFromModel(const std::array<float, FLAT_STATE_SIZE>& flatState, int validNeighborCount)
	{
#ifdef THE_FLOOR_HAS_TF_C_API
		return TensorFlowPredictor::Get().Predict(flatState, validNeighborCount);
#else
		(void)flatState;
		(void)validNeighborCount;
		return -1;
#endif
	}

	bool IsModelLoaded()
	{
#ifdef THE_FLOOR_HAS_TF_C_API
		return TensorFlowPredictor::Get().IsLoaded() || PathExists(kSavedModelPath);
#else
		return false;
#endif
	}
}

Tile::Tile(int _category, int numCategories) : category(_category), player(_category, numCategories)
{
}

int Tile::GetPower(int battleCategory)
{
	// For simplicity, we're saying a player's "Power" is 75% knowledge and 25% speed.
	return (((float)player.GetSkill(battleCategory) * 7.5f) + ((float)player.GetSpeed() * (2.5f)));
}

void Tile::AddNeighbor(Tile* neighbor)
{
	if (neighbor != this)
	{
		if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end())
		{
			neighbors.push_back(neighbor);
			neighbor->AddNeighbor(this);
		}
	}
}

void Tile::RemoveNeighbor(Tile* neighbor)
{
	auto it = std::find(neighbors.begin(), neighbors.end(), neighbor);
	
	if (it != neighbors.end())
	{
		if (!neighbor->hasLost)
		{
			neighbor->RemoveNeighbor(this);
		}
		neighbors.erase(it);
	}
}

Tile* Tile::ChooseNeighbor()
{
	hasPlayed = true;

	MLState state = GetMLState();
	std::array<float, FLAT_STATE_SIZE> flatState = state.Flatten();

	int action = PredictActionFromModel(flatState, static_cast<int>(neighbors.size()));

	if (action < 0 || action >= static_cast<int>(neighbors.size()))
	{
		action = (std::rand() % neighbors.size());
	}

	StoreDecision(state, action);

	neighbors[action]->hasPlayed = true;

	return neighbors[action];
}

void Tile::WinBattle(bool wasAttacker, Tile* loserTile)
{
	if (wasAttacker)
	{

	}
	else
	{
		category = loserTile->GetCategory();
	}

	for (int i = 0; i < loserTile->neighbors.size(); i++)
	{
		AddNeighbor(loserTile->neighbors[i]);
	}

	size++;
}

void Tile::LoseBattle()
{
	hasLost = true;

	auto it = neighbors.begin();

	while (it != neighbors.end())
	{
		(*it)->RemoveNeighbor(this);
		it++;
	}

	neighbors.clear();
	size = 0;
}

MLState Tile::GetMLState()
{
	MLState state;

	for (Tile* neighbor : neighbors)
	{
		NeighborState neighborState;
		const int battleCategory = neighbor->GetCategory();
		const float myBattlePower = GetPower(battleCategory);

		neighborState.myPower = myBattlePower;
		neighborState.enemySpeed = neighbor->hasPlayed ? neighbor->GetPlayer()->GetSpeed() : -10;
		neighborState.mySize = size;
		neighborState.enemySize = neighbor->size;
		neighborState.isChallengeable = 0.0f;

		if (neighbor->hasPlayed)
		{
			const float enemyBattlePower = neighbor->GetPower(battleCategory);
			neighborState.isChallengeable = myBattlePower >= enemyBattlePower ? 1.0f : 0.0f;
		}

		state.neighbors.push_back(neighborState);
	}

	return state;
}

void Tile::StoreDecision(MLState& state, int action)
{
	lastState = state;
	lastAction = action;
	hasPendingDecision = true;
}

void Tile::ResolveDecision(float reward, bool done)
{
	if (!hasPendingDecision)
	{
		return;
	}

	MLLogger::Log(lastState, lastAction, reward, done);
	hasPendingDecision = false;
}

StayGoState Tile::GetStayGoState()
{
	StayGoState state;

	state.values.fill(0.0f);

	MLState mlState = GetMLState();
	const std::array<float, FLAT_STATE_SIZE> flatState = mlState.Flatten();

	float challengeableCount = 0.0f;
	float knownEnemySpeedCount = 0.0f;

	for (int i = 0; i < MAX_NEIGHBORS; ++i)
	{
		const int baseIndex = i * FEATURES_PER_NEIGHBOR;
		if (flatState[baseIndex] <= 0.0f)
		{
			continue;
		}

		if (flatState[baseIndex + 4] > 0.5f)
		{
			challengeableCount += 1.0f;
		}

		if (flatState[baseIndex + 1] > -9.5f)
		{
			knownEnemySpeedCount += 1.0f;
		}
	}

	state.values[0] = static_cast<float>(neighbors.size());
	state.values[1] = static_cast<float>(size);
	state.values[2] = challengeableCount;
	state.values[3] = knownEnemySpeedCount;
	state.values[4] = flatState[0]; // first-neighbor power snapshot
	state.values[5] = flatState[3]; // first-neighbor size snapshot
	state.values[6] = static_cast<float>(player.GetSpeed());
	state.values[7] = hasPlayed ? 1.0f : 0.0f;

	return state;
}

bool Tile::IsNeuralNetLoaded()
{
	return IsModelLoaded();
}

bool Tile::ChooseStayOnFloor()
{
	if (!IsNeuralNetLoaded())
	{
		return false;
	}

	StayGoState stayGoState = GetStayGoState();
	std::array<float, FLAT_STATE_SIZE> nnState{};
	nnState.fill(0.0f);

	for (int i = 0; i < STAYGO_STATE_SIZE; ++i)
	{
		nnState[i] = stayGoState.values[i];
	}

	const int action = PredictActionFromModel(nnState, 2);
	return action == 1;
}
